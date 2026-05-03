from collections import defaultdict
import matplotlib.pyplot as plt
import os

directory = './logs'
files = filter(lambda s: s.endswith('.txt'), os.listdir(directory))
log = os.path.join(directory, tuple(files)[-1])
print(f"loaded log: {log}")

with open(log, 'r', encoding='utf-8') as f:
    log_plots = defaultdict(list)
    while line := next(f).split(':')[0]:
        if line not in log_plots:
            log_plots.update({line: []})
        else:
            f.seek(0)
            break
    for i, line in enumerate(f.readlines()):
        splits = line.split(':')
        key = splits[0]
        for s in splits[1:]:
            try:
                log_plots[key].append(eval(s))
            except Exception:
                pass

fig1 = plt.figure(1)
ax1 = fig1.add_subplot(1, 1, 1)
ax1.set_title('loss')

fig2 = plt.figure(2)
ax2 = fig2.add_subplot(1, 1, 1)
ax2.set_title('metric')

x = str('')
for k in log_plots:
    if 'epoch' in k.lower():
        x = k
        continue

    if 'loss' in k:
        ax1.plot(log_plots[x], log_plots[k], label=k)

        ax1.legend()
        ax1.grid(True, alpha=0.5)
        if min(log_plots[k]) < 1e-3:
            ax1.semilogy()

    if 'metric' in k:
        ax2.plot(log_plots[x], log_plots[k], label=k)

        ax2.legend()
        ax2.grid(True, alpha=0.5)

plt.show()