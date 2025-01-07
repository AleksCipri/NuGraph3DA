import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import Isomap
from sklearn.preprocessing import LabelEncoder
import torch

class IsomapCombinedPlot:
    def __init__(self, n_neighbors=5, n_components=2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components
        self.dataset1 = []
        self.labels1 = []
        self.dataset2 = []
        self.labels2 = []

    def update(self, x1, y1, x2, y2):
        """
        Accumulates batches of data for Isomap plotting.

        Parameters:
            x1: torch.Tensor
                Batch of source dataset features.
            y1: torch.Tensor
                Batch of source dataset labels.
            x2: torch.Tensor
                Batch of target dataset features.
            y2: torch.Tensor
                Batch of target dataset labels.
        """
        self.dataset1.append(x1.cpu().detach())
        self.labels1.append(y1.cpu().detach())
        self.dataset2.append(x2.cpu().detach())
        self.labels2.append(y2.cpu().detach())

    def compute(self):
        """
        Concatenates all accumulated batches and returns the full dataset and labels.
        """
        dataset1 = torch.cat(self.dataset1, dim=0)
        labels1 = torch.cat(self.labels1, dim=0)
        dataset2 = torch.cat(self.dataset2, dim=0)
        labels2 = torch.cat(self.labels2, dim=0)
        return dataset1, labels1, dataset2, labels2

    def reset(self):
        """
        Clears the accumulated data for the next epoch or process.
        """
        self.dataset1 = []
        self.labels1 = []
        self.dataset2 = []
        self.labels2 = []

    def plot_isomap_combined_concatenated(self, dataset1, labels1, dataset2, labels2, epoch=None, class_names=None):
        """
        Plots Isomap embeddings of two datasets with custom class names in the legend.
        """
        # Convert datasets and labels to numpy
        dataset1_np = dataset1.cpu().detach().numpy()
        labels1_np = labels1.cpu().detach().numpy()
        dataset2_np = dataset2.cpu().detach().numpy()
        labels2_np = labels2.cpu().detach().numpy()

        # Concatenate datasets and labels
        combined_data = np.concatenate([dataset1_np, dataset2_np], axis=0)
        combined_labels = np.concatenate([labels1_np, labels2_np])

        # Fit Isomap
        isomap = Isomap(n_neighbors=self.n_neighbors, n_components=self.n_components)
        combined_embedding = isomap.fit_transform(combined_data)

        # Split embeddings back into separate datasets
        embedding1 = combined_embedding[:len(dataset1)]
        embedding2 = combined_embedding[len(dataset1):]

        # Ensure class names list is provided and valid
        if class_names is None:
            class_names = [f"Class {i}" for i in range(len(np.unique(combined_labels)))]
        elif len(class_names) <= max(combined_labels):
            raise ValueError("The class_names list must include a name for each class index in the data.")

        #Generate a colormap for all unique classes
        unique_classes = np.unique(combined_labels)
        colors = plt.cm.inferno(np.linspace(0, 1, len(unique_classes)))
        class_color_map = {cls: colors[idx] for idx, cls in enumerate(unique_classes)}
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 10))

        # Plot dataset1 with circles
        for cls in np.unique(labels1_np):
            mask = labels1_np == cls
            ax.scatter(
                embedding1[mask, 0], embedding1[mask, 1],
                label=f"Source - {class_names[cls]}", color=class_color_map[cls],
                marker='o', s=100, edgecolor='k', alpha=0.8
            )

        # Plot dataset2 with triangles
        for cls in np.unique(labels2_np):
            mask = labels2_np == cls
            ax.scatter(
                embedding2[mask, 0], embedding2[mask, 1],
                label=f"Target - {class_names[cls]}", color=class_color_map[cls],
                marker='^', s=150, edgecolor='k', alpha=0.8
            )

        # Add legend and labels
        ax.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1, 1), framealpha=1.0)
        title = "Isomap Embeddings of Source and Target Datasets"
        if epoch is not None:
            title += f" (Epoch {epoch})"
        ax.set_title(title, fontsize=16)

        # Close the figure after creation to avoid reuse
        plt.close(fig)

        return fig