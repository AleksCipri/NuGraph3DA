import torch
import torch.nn as nn

class SemanticAlignmentLoss(nn.Module):
    def __init__(self, metric='euclidean'):
        """
        Semantic Alignment Loss.

        Parameters:
            metric: str, optional (default='euclidean')
                The distance metric to use ('euclidean' or 'cosine').
        """
        super(SemanticAlignmentLoss, self).__init__()
        self.metric = metric

    def forward(self, features1, labels1, features2, labels2):
        """
        Computes the semantic alignment loss.

        Parameters:
            features1: torch.Tensor, shape (batch_size1, feature_dim)
                Feature embeddings from dataset 1.
            labels1: torch.Tensor, shape (batch_size1,)
                Labels for dataset 1.
            features2: torch.Tensor, shape (batch_size2, feature_dim)
                Feature embeddings from dataset 2.
            labels2: torch.Tensor, shape (batch_size2,)
                Labels for dataset 2.

        Returns:
            torch.Tensor: The semantic alignment loss.
        """
        unique_classes = torch.unique(torch.cat([labels1, labels2]))
        total_loss = 0.0
        count = 0

        for cls in unique_classes:
            # Select features for the current class
            cls_features1 = features1[labels1 == cls]
            cls_features2 = features2[labels2 == cls]

            if cls_features1.size(0) == 0 or cls_features2.size(0) == 0:
                continue

            # Compute pairwise distances
            if self.metric == 'euclidean':
                distances = torch.cdist(cls_features1, cls_features2, p=2)  # Euclidean distance
            elif self.metric == 'cosine':
                cls_features1 = nn.functional.normalize(cls_features1, dim=1)
                cls_features2 = nn.functional.normalize(cls_features2, dim=1)
                distances = 1 - torch.mm(cls_features1, cls_features2.T)  # Cosine similarity as distance
            else:
                raise ValueError("Unsupported metric. Use 'euclidean' or 'cosine'.")

            # Average distance for this class
            total_loss += distances.mean()
            count += 1

        return total_loss / count if count > 0 else torch.tensor(0.0, requires_grad=True)
