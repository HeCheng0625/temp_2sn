import torch

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.manifold import TSNE

# Load some data
train_spk_ids = torch.load('train_spk_ids.pt')
test_spk_ids = torch.load('test_spk_ids.pt')
train_spk_embed = torch.load('train_spk_embed.pt')  # 20 * 512
test_spk_embed = torch.load('test_spk_embed.pt') # 20 * 512
X = torch.cat([train_spk_embed, test_spk_embed], 0).cpu().numpy() # 40 * 512
train_labels = [f'train_{int(i)}'  for i in train_spk_ids]
test_labels = [f'test_{int(i)}' for i in test_spk_ids]

Y = np.array(train_labels + test_labels, dtype=str)
# print(X.shape, Y)
# exit()
# Apply t-SNE
tsne = TSNE(n_components=2, random_state=0) # create a TSNE object
X_2d = tsne.fit_transform(X) # transform the features to 2D
print(X_2d.shape)
# exit()
# Plot the result
plt.figure(figsize=(6, 5))
# colors = plt.cm.rainbow(np.linspace(0, 1, len(Y))) # create a color map
labels = set(Y) # get the unique labels

for i, label in enumerate(labels):
    plt.scatter(X_2d[Y == label, 0], X_2d[Y == label, 1], label=label) # plot each label with a different color

plt.legend()
plt.title('t-SNE visualization of the speaker embedding')
plt.savefig('spk_emb.png')
plt.show()
