

from brian2 import *
# import matplotlib.pyplot as plt

start_scope()

eqs = '''
dv/dt = (I-v)/tau : 1
I : 1
tau : second
'''
G = NeuronGroup(3, eqs, threshold='v>1', reset='v = 0', method='exact')
G.I = [2, 0, 0]
G.tau = [10, 100, 100]*ms

# Comment these two lines out to see what happens without Synapses
S = Synapses(G, G, 'w : 1', on_pre='v_post += w')
S.connect(i=0, j=[1, 2])
S.w = 'j*0.2'

M = StateMonitor(G, 'v', record=True)

run(50*ms)

figure()
plot(M.t/ms, M.v[0], label='Neuron 0')
plot(M.t/ms, M.v[1], label='Neuron 1')
plot(M.t/ms, M.v[2], label='Neuron 2')
xlabel('Time (ms)')
ylabel('v')
legend();

show()
#%%
from brian2 import *
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")
start_scope()

N = 6
G = NeuronGroup(N, 'v:1')
S = Synapses(G, G)
S.connect(i=[0, 1, 2, 3, 4, 5],
            j=[1, 2, 3, 4, 5,0])

Ns = 6
Nt = 6
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(zeros(Ns), arange(Ns), 'ok', ms=10)
plt.plot(ones(Nt), arange(Nt), 'ok', ms=10)
for i, j in zip(S.i, S.j):
    plt.plot([0, 1], [i, j], '-k')
plt.xticks([0, 1], ['Source', 'Target'])
plt.ylabel('Neuron index')
plt.xlim(-0.1, 1.1)
plt.ylim(-1, max(Ns, Nt))
plt.subplot(122)
plt.plot(S.i[:], S.j[:], 'ok')
plt.xlim(-1, Ns)
plt.ylim(-1, Nt)
plt.xlabel('Source neuron index')
plt.ylabel('Target neuron index')
plt.show()

#%%
import matplotlib.pyplot as plt
plt.figure()

plt.plot(range(6))
plt.show()
#%%
plt.figure()

plt.plot(range(100))
plt.show()