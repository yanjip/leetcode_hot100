import torch
import torch.nn as nn
import torch.nn.functional as F
import unittest

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                       Shape:
                         - binary/multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                         - binary: (batch_size,)
                         - multi-label: (batch_size, num_classes)
                         - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = F.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

class TestFocalLoss(unittest.TestCase):

    def test_binary_focal_loss(self):
        """ Test the FocalLoss with binary classification. """
        criterion = FocalLoss(gamma=2, alpha=0.25, task_type='binary')
        inputs = torch.randn(16)  # Logits from the model (batch_size=16)
        targets = torch.randint(0, 2, (16,)).float()  # Binary ground truth (0 or 1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_class_focal_loss(self):
        """ Test the FocalLoss with multi-class classification. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, alpha=[0.25] * num_classes, task_type='multi-class', num_classes=num_classes)
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, num_classes, (16,))  # Ground truth with integer class labels (0 to num_classes-1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_label_focal_loss(self):
        """ Test the FocalLoss with multi-label classification. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, alpha=0.25, task_type='multi-label')
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, 2, (16, num_classes)).float()  # Multi-label ground truth (0 or 1 for each class)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_binary_focal_loss_no_alpha(self):
        """ Test the FocalLoss with binary classification without alpha. """
        criterion = FocalLoss(gamma=2, task_type='binary')
        inputs = torch.randn(16)  # Logits from the model (batch_size=16)
        targets = torch.randint(0, 2, (16,)).float()  # Binary ground truth (0 or 1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_class_focal_loss_no_alpha(self):
        """ Test the FocalLoss with multi-class classification without alpha. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, task_type='multi-class', num_classes=num_classes)
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, num_classes, (16,))  # Ground truth with integer class labels (0 to num_classes-1)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_multi_label_focal_loss_no_alpha(self):
        """ Test the FocalLoss with multi-label classification without alpha. """
        num_classes = 5
        criterion = FocalLoss(gamma=2, task_type='multi-label')
        inputs = torch.randn(16, num_classes)  # Logits from the model (batch_size=16, num_classes=5)
        targets = torch.randint(0, 2, (16, num_classes)).float()  # Multi-label ground truth (0 or 1 for each class)

        # Calculate the focal loss
        loss = criterion(inputs, targets)

        # Assert the loss is a scalar and positive
        self.assertTrue(loss.item() >= 0)
        self.assertTrue(loss.dim() == 0)  # Scalar output

    def test_invalid_task_type(self):
        """ Test FocalLoss with an invalid task type """
        with self.assertRaises(ValueError):
            criterion = FocalLoss(gamma=2, task_type='invalid-task')
            inputs = torch.randn(16, 5)
            targets = torch.randint(0, 5, (16,))
            criterion(inputs, targets)

if __name__ == '__main__':
    unittest.main()