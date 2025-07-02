"""
Standard computer vision baselines for image classification comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
import numpy as np
import logging
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import time
import os
import sys
from PIL import Image

from examples.vision.image_utils import extract_features_from_loader

logger = logging.getLogger(__name__)


class BaseImageClassifier:
    """Base class for image classifiers."""
    
    def __init__(self, num_classes: int, class_names: Optional[List[str]] = None):
        self.num_classes = num_classes
        self.class_names = class_names
        self.is_fitted = False
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> 'BaseImageClassifier':
        raise NotImplementedError
        
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        raise NotImplementedError
        
    def evaluate(self, test_loader: DataLoader, test_labels: np.ndarray) -> Dict[str, Any]:
        """Evaluate classifier on test data."""
        start_time = time.time()
        predictions = self.predict(test_loader)
        
        accuracy = accuracy_score(test_labels, predictions)
        
        results = {
            'accuracy': accuracy,
            'prediction_time': time.time() - start_time,
            'num_test_samples': len(test_labels),
            'predictions': predictions,
            'true_labels': test_labels
        }
        
        if self.class_names:
            results['classification_report'] = classification_report(
                test_labels, predictions,
                target_names=self.class_names,
                output_dict=True
            )
        
        results['confusion_matrix'] = confusion_matrix(test_labels, predictions)
        
        return results


class ResNetClassifier(BaseImageClassifier):
    """ResNet-50 fine-tuned classifier."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        pretrained: bool = True,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        device: str = 'cuda'
    ):
        super().__init__(num_classes, class_names)
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Initialize model
        self.model = models.resnet50(pretrained=pretrained)
        
        # Replace final layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> 'ResNetClassifier':
        """Fine-tune ResNet-50 on training data."""
        logger.info(f"Fine-tuning ResNet-50 for {self.num_epochs} epochs")
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # Validation accuracy
            val_accuracy = self._evaluate_loader(val_loader)
            
            train_accuracy = correct_predictions / total_samples
            avg_loss = running_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}: "
                       f"Loss: {avg_loss:.4f}, "
                       f"Train Acc: {train_accuracy:.4f}, "
                       f"Val Acc: {val_accuracy:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict labels for test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def _evaluate_loader(self, data_loader: DataLoader) -> float:
        """Evaluate model on a data loader."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.model.train()
        return correct / total


class EfficientNetClassifier(BaseImageClassifier):
    """EfficientNet-B0 fine-tuned classifier."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        pretrained: bool = True,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        device: str = 'cuda'
    ):
        super().__init__(num_classes, class_names)
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Initialize model
        self.model = models.efficientnet_b0(pretrained=pretrained)
        
        # Replace final layer
        self.model.classifier[1] = nn.Linear(self.model.classifier[1].in_features, num_classes)
        self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> 'EfficientNetClassifier':
        """Fine-tune EfficientNet-B0 on training data."""
        logger.info(f"Fine-tuning EfficientNet-B0 for {self.num_epochs} epochs")
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # Validation accuracy
            val_accuracy = self._evaluate_loader(val_loader)
            
            train_accuracy = correct_predictions / total_samples
            avg_loss = running_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}: "
                       f"Loss: {avg_loss:.4f}, "
                       f"Train Acc: {train_accuracy:.4f}, "
                       f"Val Acc: {val_accuracy:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict labels for test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def _evaluate_loader(self, data_loader: DataLoader) -> float:
        """Evaluate model on a data loader."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.model.train()
        return correct / total


class ViTClassifier(BaseImageClassifier):
    """Vision Transformer fine-tuned classifier."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        pretrained: bool = True,
        learning_rate: float = 0.001,
        num_epochs: int = 10,
        device: str = 'cuda'
    ):
        super().__init__(num_classes, class_names)
        self.pretrained = pretrained
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.device = device
        
        # Initialize model
        self.model = models.vit_b_16(pretrained=pretrained)
        
        # Replace final layer
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
        self.model.to(device)
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> 'ViTClassifier':
        """Fine-tune Vision Transformer on training data."""
        logger.info(f"Fine-tuning Vision Transformer for {self.num_epochs} epochs")
        
        self.model.train()
        
        for epoch in range(self.num_epochs):
            running_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_samples += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
            
            # Validation accuracy
            val_accuracy = self._evaluate_loader(val_loader)
            
            train_accuracy = correct_predictions / total_samples
            avg_loss = running_loss / len(train_loader)
            
            logger.info(f"Epoch {epoch+1}/{self.num_epochs}: "
                       f"Loss: {avg_loss:.4f}, "
                       f"Train Acc: {train_accuracy:.4f}, "
                       f"Val Acc: {val_accuracy:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict labels for test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
            
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
    
    def _evaluate_loader(self, data_loader: DataLoader) -> float:
        """Evaluate model on a data loader."""
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in data_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        self.model.train()
        return correct / total


class DINOV2LinearProbe(BaseImageClassifier):
    """Linear probe on frozen DINOV2 features."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        dinov2_model: str = "dinov2_vitb14",
        device: str = 'cuda'
    ):
        super().__init__(num_classes, class_names)
        self.dinov2_model_name = dinov2_model
        self.device = device
        
        # Load DINOV2 model
        from marvis.data.embeddings import load_dinov2_model
        self.feature_extractor = load_dinov2_model(dinov2_model, device)
        
        # Initialize linear classifier
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(max_iter=1000, random_state=42)
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> 'DINOV2LinearProbe':
        """Fit linear probe on DINOV2 features."""
        logger.info("Extracting DINOV2 features and fitting linear probe")
        
        # Extract features
        train_features, train_labels = extract_features_from_loader(
            self.feature_extractor, train_loader, self.device
        )
        val_features, val_labels = extract_features_from_loader(
            self.feature_extractor, val_loader, self.device
        )
        
        # Standardize features
        train_features_scaled = self.scaler.fit_transform(train_features)
        val_features_scaled = self.scaler.transform(val_features)
        
        # Fit classifier
        self.classifier.fit(train_features_scaled, train_labels)
        
        # Log validation accuracy
        val_accuracy = self.classifier.score(val_features_scaled, val_labels)
        logger.info(f"DINOV2 Linear Probe validation accuracy: {val_accuracy:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict labels for test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        test_features, _ = extract_features_from_loader(
            self.feature_extractor, test_loader, self.device
        )
        
        # Standardize and predict
        test_features_scaled = self.scaler.transform(test_features)
        predictions = self.classifier.predict(test_features_scaled)
        
        return predictions


class DINOV2RandomForest(BaseImageClassifier):
    """Random Forest on DINOV2 features."""
    
    def __init__(
        self,
        num_classes: int,
        class_names: Optional[List[str]] = None,
        dinov2_model: str = "dinov2_vitb14",
        n_estimators: int = 100,
        device: str = 'cuda'
    ):
        super().__init__(num_classes, class_names)
        self.dinov2_model_name = dinov2_model
        self.n_estimators = n_estimators
        self.device = device
        
        # Load DINOV2 model
        from marvis.data.embeddings import load_dinov2_model
        self.feature_extractor = load_dinov2_model(dinov2_model, device)
        
        # Initialize Random Forest classifier
        self.scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> 'DINOV2RandomForest':
        """Fit Random Forest on DINOV2 features."""
        logger.info("Extracting DINOV2 features and fitting Random Forest")
        
        # Extract features
        train_features, train_labels = extract_features_from_loader(
            self.feature_extractor, train_loader, self.device
        )
        val_features, val_labels = extract_features_from_loader(
            self.feature_extractor, val_loader, self.device
        )
        
        # Standardize features
        train_features_scaled = self.scaler.fit_transform(train_features)
        val_features_scaled = self.scaler.transform(val_features)
        
        # Fit classifier
        self.classifier.fit(train_features_scaled, train_labels)
        
        # Log validation accuracy
        val_accuracy = self.classifier.score(val_features_scaled, val_labels)
        logger.info(f"DINOV2 Random Forest validation accuracy: {val_accuracy:.4f}")
        
        self.is_fitted = True
        return self
    
    def predict(self, test_loader: DataLoader) -> np.ndarray:
        """Predict labels for test data."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Extract features
        test_features, _ = extract_features_from_loader(
            self.feature_extractor, test_loader, self.device
        )
        
        # Standardize and predict
        test_features_scaled = self.scaler.transform(test_features)
        predictions = self.classifier.predict(test_features_scaled)
        
        return predictions