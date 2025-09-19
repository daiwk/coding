# coding=utf-8
# Copyright 2024 The Efficientids Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Clustering utilities.

This provides a utility class to calculate Kmeans, random and frequency
clustering from a set of embeddings and maintain the clustering state.
It also provides a method to calculate the cluster means for a given clustering
which can be used as the embeddings to represent the clusters.

Underneath, this uses sklearn.cluster.KMeans for k-means clustering.
Frequency clustering expects the vocabulary to be sorted by frequency, with the
most frequent item having the lowest id.

"""

import numpy as np
from sklearn.cluster import KMeans
import sklearn
from sklearn.cluster import SpectralClustering
import collections
from collections.abc import Sequence
import pickle
import re

from absl import app
import jax
import jax.numpy as jnp
import sklearn.cluster
import tensorflow.compat.v2 as tf


ClusterState = collections.namedtuple(
    'ClusterState', ['cluster_assignments', 'cluster_indices', 'in_cluster_id']
)


class ClusteringStateManager:
  """Manages the clustering state for InterleavedTransformerLm."""

  def __init__(self, vocab_size, num_clusters=None, max_cluster_size=1000):
    self.vocab_size = vocab_size

    if num_clusters is None:
      self.num_clusters = jnp.ceil(jnp.sqrt(vocab_size)).astype(jnp.int32)
    else:
      self.num_clusters = num_clusters

    self.clustering_state = ClusterState(
        0.0,
        0.0,
        0.0,
    )
    self.cluster_means = None
    self.max_cluster_size = max_cluster_size

  def recalculate_clusters(self, mode='random', shortlist=0, params=None):
    """Recompute clustering using a particular clustering algorithm.

    This method recomputes the clustering of the vocabulary, according to a
    particular clustering criterion. In addition, the top k words in the
    vocabulary can be assigned their own cluster with the shortlist argument.

    Supported clustering modes:
      random: randomly compute a clustering
      frequency: clusters the vocabulary by frequency (assumes the ids are
                 sorted by frequency, most frequent with the lowest id)
      k-means: performs a k-means clustering of passed in input parameters

    Args:
      mode: the clustering mode
      shortlist: how many vocab items to give their own cluster.
      params: the embeddings to cluster with
    """

    if shortlist >= self.num_clusters:
      raise ValueError('shortlist has to be less than num_clusters')

    shortlist_clusters = jnp.arange(shortlist)

    rest = None
    if mode == 'random':
      rest = jax.random.choice(
          key=jax.random.PRNGKey(0),
          a=jnp.arange(shortlist, self.num_clusters),
          shape=(self.vocab_size - shortlist,),
      )

    elif mode == 'frequency':
      clusters_left = self.num_clusters - shortlist
      indices_to_sort = self.vocab_size - shortlist
      bins = jnp.tile(jnp.arange(shortlist, shortlist + clusters_left), int(jnp.ceil(indices_to_sort / clusters_left)))[:indices_to_sort]
      rest = bins + shortlist

    elif mode == 'spectral':
      assert params is not None
      kmeans = sklearn.cluster.SpectralClustering(
          n_clusters=self.num_clusters - shortlist, affinity='nearest_neighbors', n_neighbors=50).fit(params[shortlist:])
      labels = kmeans.labels_
      print(np.bincount(labels))

      while any(np.bincount(labels) > self.max_cluster_size):
        new_labels = labels.copy()
        cluster_sizes = np.bincount(labels)
        for cluster_id in np.unique(labels):
            if cluster_sizes[cluster_id] > self.max_cluster_size:
                # Split the cluster using K-means
                cluster_data = params[shortlist:][labels == cluster_id]
                sub_labels = sklearn.cluster.SpectralClustering(n_clusters=2, affinity='nearest_neighbors', n_neighbors=50).fit_predict(cluster_data)
                sub_labels[sub_labels == 0] = np.max(new_labels) + 1
                sub_labels[sub_labels == 1] = cluster_id
                new_labels[labels == cluster_id] = sub_labels
        labels = new_labels
      bins = labels
      print(np.bincount(labels))


      # We shift the assignments to take into account the shortlist
      rest = bins + shortlist

    elif mode == 'hierarchical':
      assert params is not None
      hierarchical = sklearn.cluster.AgglomerativeClustering(
          n_clusters=self.num_clusters - shortlist).fit(params[shortlist:])
      bins = hierarchical.labels_

      # We shift the assignments to take into account the shortlist
      rest = bins + shortlist
    print('rest:')
    print(rest)

    cluster_assignments = jnp.concatenate((shortlist_clusters, rest))

    total_cluster_count = len(np.bincount(rest))
    indices, in_cluster_id = self._fill_in_cluster_data(cluster_assignments, total_cluster_count)

    self.clustering_state = ClusterState(
        cluster_assignments=cluster_assignments,
        cluster_indices=indices,
        in_cluster_id=in_cluster_id,
    )

  def cluster_ratio(self, shortlist):
    """Returns max_cluster_size / min_cluster_size."""
    min_cluster_size = min(
        (self.clustering_state.cluster_indices != -1).sum(-1)[shortlist:]
    )

    return self.max_cluster_size / min_cluster_size

  def calculate_cluster_means(self, params):
    """Given word embeddings and a clustering, calculates cluster means."""

    mask = self.clustering_state.cluster_indices != -1

    cluster_embeddings = jnp.where(
        mask[Ellipsis, None], params[self.clustering_state.cluster_indices], 0.0
    )

    cluster_means = cluster_embeddings.sum(1) / mask.sum(-1)[:, None]

    return cluster_means

  def _fill_in_cluster_data(self, cluster_assignments, total_cluster_count):
    """Given cluster assignments, calculate other data for the clustering."""

    # Calculate the cluster->vocab mappings. We use argsort with searchsorted
    # to return the fencepost indices, and then np.split to avoid python loops
    sorted_indices = jnp.argsort(cluster_assignments)
    fencepost_indices = jnp.searchsorted(
        cluster_assignments[sorted_indices], jnp.arange(total_cluster_count)
    )

    indices_for_clusters = jnp.split(sorted_indices, fencepost_indices)[1:]

    for x in indices_for_clusters:
      print(len(x))
    max_cluster_size = max(len(x) for x in indices_for_clusters)


    # For efficiency reasons, pre-pad and store this as a single array.
    # -1 indicates padding

    padded_indices = [
        jnp.pad(
            x,
            (0, max_cluster_size - len(x)),
            mode='constant',
            constant_values=-1,
        )
        for x in indices_for_clusters
    ]

    indices = jnp.stack(padded_indices, 0)

    # Also precalcuate the id within the cluster for each vocab item
    in_cluster_id = jnp.array(
        [jnp.where(indices == x)[1] for x in jnp.arange(self.vocab_size)]
    )[:, 0]

    return indices, in_cluster_id


def process_embeddings(item_embedding_size, vocab_size, item_embedding_raw):
  """Process embeddings."""
  embeddings = jnp.zeros((vocab_size, item_embedding_size), dtype=jnp.float32)
  with tf.io.gfile.GFile(
      item_embedding_raw,
      'r',
  ) as f:
    for idx, line in enumerate(f):
      line = re.sub(r'\s+', ' ', line).strip()
      embedding = [float(x) for x in line.split(' ')]
      embeddings = embeddings.at[idx].set(embedding)
  return embeddings



def main(argv):
  del argv
  cluster_model = 'kmeans'
  item_vocab_size = 17388
  item_embedding_size = 512
  num_item_clusters = 100
  raw_embeddings_file: str = (
      '.../office_item.csv'
  )
  output_pkl: str = (
      '.../office_kmeans.pickle'
  )
  item_embeddings = process_embeddings(
      item_embedding_size, item_vocab_size, raw_embeddings_file
  )
  cluster_manager = ClusteringStateManager(item_vocab_size, num_item_clusters)
  cluster_manager.recalculate_clusters(
      mode=cluster_model, params=item_embeddings
  )
  cluster_means = cluster_manager.calculate_cluster_means(item_embeddings)
  with tf.io.gfile.GFile(
      output_pkl,
      'wb',
  ) as f:
    pickle.dump(cluster_manager.clustering_state.cluster_assignments, f)
    pickle.dump(cluster_manager.clustering_state.cluster_indices, f)
    pickle.dump(cluster_manager.clustering_state.in_cluster_id, f)
    pickle.dump(cluster_means, f)


if __name__ == '__main__':
  app.run(main)
