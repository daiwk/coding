# Re-execute after state reset
from graphviz import Digraph

g = Digraph('Distill', format='png')
g.attr(rankdir='LR', splines='spline', nodesep='0.6', ranksep='0.6')
g.attr('node', shape='box', fontsize='11', style='rounded,filled', color='#222222')

# Input
g.node('inp', 'Input:\nquery + candidates', fillcolor='white')

# Teacher cluster
with g.subgraph(name='cluster_teacher') as c:
    c.attr(label='Teacher (frozen)', color='#6aa6ff')
    c.attr('node', fillcolor='#eaf3ff')
    c.node('T', 'Teacher Model')
    c.attr('node', fillcolor='white')
    c.node('Ts', 'Teacher Scores')
    c.node('Tf', 'Teacher Features')
    c.edge('T', 'Ts')
    c.edge('T', 'Tf')

# Student cluster
with g.subgraph(name='cluster_student') as c:
    c.attr(label='Student (trainable)', color='#77d38d')
    c.attr('node', fillcolor='#eaffea')
    c.node('S', 'Student Model')
    c.attr('node', fillcolor='white')
    c.node('Ss', 'Student Scores')
    c.node('Sf', 'Student Features')
    c.edge('S', 'Ss')
    c.edge('S', 'Sf')

# Loss nodes
g.attr('node', fillcolor='#fff3e0')
g.node('Llogit', 'Logit Distillation\n(KL / Listwise-KL)')
g.node('Lfeat', 'Feature Distillation\n(MSE / Cos / InfoNCE)')
g.node('Ltask', 'Task Loss\n(CE / Pairwise / Listwise)')

# Ground truth
g.attr('node', fillcolor='white')
g.node('y', 'Ground Truth Labels')

# Wiring (forward)
g.edge('inp', 'T')
g.edge('inp', 'S')

# Distillation forward
g.edge('Ts', 'Llogit', label='forward')
g.edge('Ss', 'Llogit', label='forward')
g.edge('Tf', 'Lfeat', label='forward')
g.edge('Sf', 'Lfeat', label='forward')

# Task loss forward
g.edge('y', 'Ltask', label='forward')
g.edge('Ss', 'Ltask', label='forward')

# Gradient flow (dashed to student)
g.edge('Llogit', 'S', style='dashed', color='#888888', label='grad')
g.edge('Lfeat', 'S', style='dashed', color='#888888', label='grad')
g.edge('Ltask', 'S', style='dashed', color='#888888', label='grad')

# Stop-grad notes
g.attr('node', shape='note', fontsize='9', style='filled', fillcolor='white')
g.node('SGs', 'stop-grad')
g.node('SGf', 'stop-grad')
g.edge('Ts', 'SGs', arrowhead='none', color='#999999')
g.edge('Tf', 'SGf', arrowhead='none', color='#999999')

outfile = './ranking_distillation_graphviz'
g.render(filename=outfile, cleanup=True)
outfile + '.png'

