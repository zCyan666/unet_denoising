import matplotlib.pyplot as plt
import os

log_items = {'epochs': [], 'train_loss': [], 'val_loss': []}
directory = './logs'
files = filter(lambda s: s.endswith('.txt'), os.listdir(directory))
log = os.path.join(directory, tuple(files)[-1])
print(f"loaded log: {log}")
with open(log, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f.readlines()):
        splits = line.split(':')
        it_len = len(log_items)
        for s in splits:
            try:
                if i % it_len == 0:
                    log_items['epochs'].append(eval(s))
                    continue
                if i % it_len == 1:
                    log_items['train_loss'].append(eval(s))
                    continue
                else:
                    log_items['val_loss'].append(eval(s))

            except Exception as e:
                pass

ax = plt.subplot()
ax.plot(log_items.get('epochs'), log_items.get('train_loss'), label='train loss')
ax.plot(log_items.get('epochs'), log_items.get('val_loss'), label='val loss')
ax.set_title('loss')
ax.legend()
ax.grid(True, alpha=0.5)
ax.semilogy()
plt.show()