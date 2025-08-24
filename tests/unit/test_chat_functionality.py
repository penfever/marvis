#!/usr/bin/env python
"""
Tests for MARVIS chat functionality.

This module tests the conversational interface that allows users to interact
with MARVIS predictions through natural language.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import time
import tempfile
import shutil
from pathlib import Path

# Import MARVIS modules
from marvis.models.marvis_tsne import MarvisTsneClassifier


class TestMarvisChatFunctionality:
    """Test suite for MARVIS chat functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        # Create test data
        np.random.seed(42)
        self.X_train = np.random.randn(50, 10)
        self.y_train = np.random.randint(0, 3, 50)
        self.X_test = np.random.randn(20, 10)
        self.y_test = np.random.randint(0, 3, 20)
        self.class_names = ["Class A", "Class B", "Class C"]
        
        # Create temporary directory for test outputs
        self.temp_dir = Path(tempfile.mkdtemp(prefix='marvis_chat_test_'))

    def teardown_method(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_mock_classifier(self, with_predictions=True):
        """Create a mock MARVIS classifier for testing."""
        classifier = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id="mock-model",
            tsne_perplexity=10,
            tsne_max_iter=100,
            seed=42
        )
        
        # Mock the VLM wrapper
        mock_vlm = Mock()
        mock_vlm.generate_response = Mock(return_value="This is a mock chat response.")
        classifier.vlm_wrapper = mock_vlm
        
        # Mock the logger
        classifier.logger = Mock()
        
        # Set up basic attributes
        classifier.class_names = self.class_names
        classifier.unique_classes = [0, 1, 2]
        classifier.modality = "tabular"
        classifier.effective_model_id = "mock-model"
        
        if with_predictions:
            # Simulate that predictions have been made
            classifier._last_prediction_context = {
                'task_type': 'classification',
                'class_names': self.class_names,
                'num_test_samples': len(self.X_test),
                'completed_samples': len(self.X_test),
                'completion_rate': 1.0,
                'modality': 'tabular',
                'accuracy': 0.85,
                'recent_predictions': [
                    {'predicted_class': 'Class A', 'true_class': 'Class A', 'confidence': 'N/A'},
                    {'predicted_class': 'Class B', 'true_class': 'Class C', 'confidence': 'N/A'}
                ],
                'timestamp': time.time()
            }
        
        return classifier

    def test_chat_without_predictions_raises_error(self):
        """Test that chat raises error when called before predictions."""
        classifier = self.create_mock_classifier(with_predictions=False)
        
        with pytest.raises(RuntimeError, match="Chat requires predictions to be made first"):
            classifier.chat("Hello!")

    def test_chat_basic_functionality(self):
        """Test basic chat functionality."""
        classifier = self.create_mock_classifier()
        
        response = classifier.chat("What were the main patterns in the data?")
        
        assert isinstance(response, str)
        assert response == "This is a mock chat response."
        classifier.vlm_wrapper.generate_response.assert_called_once()

    def test_chat_history_management(self):
        """Test chat history is properly maintained."""
        classifier = self.create_mock_classifier()
        
        # First chat
        response1 = classifier.chat("First question")
        assert len(classifier.get_chat_history()) == 1
        
        # Second chat
        response2 = classifier.chat("Second question")
        assert len(classifier.get_chat_history()) == 2
        
        # Check history structure
        history = classifier.get_chat_history()
        assert history[0]['user'] == "First question"
        assert history[0]['assistant'] == "This is a mock chat response."
        assert 'timestamp' in history[0]
        
        assert history[1]['user'] == "Second question"
        assert history[1]['assistant'] == "This is a mock chat response."

    def test_chat_context_includes_prediction_info(self):
        """Test that chat context includes prediction information."""
        classifier = self.create_mock_classifier()
        
        classifier.chat("Tell me about the results")
        
        # Check that the VLM was called with context including prediction info
        call_args = classifier.vlm_wrapper.generate_response.call_args
        prompt = call_args[1]['text_input']  # keyword argument
        
        assert "MARVIS Classification Context" in prompt
        assert "Modality: tabular" in prompt
        assert "Accuracy: 0.85" in prompt
        assert "Class A" in prompt
        assert "Class B" in prompt
        assert "Class C" in prompt

    def test_chat_max_history_limit(self):
        """Test that chat history respects maximum history limit."""
        classifier = self.create_mock_classifier()
        
        # Add many chat exchanges
        for i in range(15):
            classifier.chat(f"Question {i}")
        
        # Test with max_history=5
        classifier.vlm_wrapper.generate_response.reset_mock()
        classifier.chat("Final question", max_history=5)
        
        # Check that only recent history is included
        call_args = classifier.vlm_wrapper.generate_response.call_args
        prompt = call_args[1]['text_input']
        
        # Should include recent questions but not all 15
        assert "Question 14" in prompt
        assert "Question 13" in prompt
        assert "Question 10" in prompt  # Should be included (within last 5)
        assert "Question 5" not in prompt  # Should not be included

    def test_clear_chat_history(self):
        """Test clearing chat history."""
        classifier = self.create_mock_classifier()
        
        # Add some history
        classifier.chat("First question")
        classifier.chat("Second question")
        assert len(classifier.get_chat_history()) == 2
        
        # Clear history
        classifier.clear_chat_history()
        assert len(classifier.get_chat_history()) == 0

    def test_chat_with_vlm_error_handling(self):
        """Test chat error handling when VLM fails."""
        classifier = self.create_mock_classifier()
        classifier.vlm_wrapper.generate_response.side_effect = Exception("VLM error")
        
        response = classifier.chat("This will cause an error")
        
        assert "I apologize, but I encountered an error" in response
        assert "VLM error" in response
        
        # History should still be maintained even with errors
        history = classifier.get_chat_history()
        assert len(history) == 1
        assert history[0]['user'] == "This will cause an error"
        assert "error" in history[0]['assistant']

    def test_chat_without_vlm_wrapper_loads_model(self):
        """Test that chat loads VLM model if not already loaded."""
        classifier = self.create_mock_classifier()
        classifier.vlm_wrapper = None  # Simulate unloaded VLM
        
        # Mock the _load_vlm method
        mock_vlm = Mock()
        mock_vlm.generate_response = Mock(return_value="Response after loading")
        
        def mock_load_vlm():
            classifier.vlm_wrapper = mock_vlm  # Set VLM wrapper when loading
        
        with patch.object(classifier, '_load_vlm', side_effect=mock_load_vlm) as mock_load:
            response = classifier.chat("Test question")
            
            mock_load.assert_called_once()
            assert response == "Response after loading"

    def test_chat_context_with_evaluation_results(self):
        """Test chat context when evaluation results are available."""
        classifier = self.create_mock_classifier()
        
        # Add evaluation-specific context
        classifier._last_prediction_context.update({
            'accuracy': 0.92,
            'balanced_accuracy': 0.89,
            'f1_macro': 0.91,
            'evaluation_completed': True
        })
        
        classifier.chat("How well did the model perform?")
        
        call_args = classifier.vlm_wrapper.generate_response.call_args
        prompt = call_args[1]['text_input']
        
        assert "Accuracy: 0.92" in prompt

    def test_chat_prompt_structure(self):
        """Test that chat prompt has correct structure and content."""
        classifier = self.create_mock_classifier()
        
        classifier.chat("Explain the visualization")
        
        call_args = classifier.vlm_wrapper.generate_response.call_args
        prompt = call_args[1]['text_input']
        
        # Check prompt structure
        assert "You are MARVIS" in prompt
        assert "Dataset Information:" in prompt
        assert "Model Configuration:" in prompt
        assert "Recent Predictions Summary:" in prompt
        assert "Recent Prediction Examples:" in prompt
        assert "Current Question:" in prompt
        assert "Explain the visualization" in prompt

    def test_chat_with_different_modalities(self):
        """Test chat context for different modalities."""
        # Test with vision modality
        vision_classifier = MarvisTsneClassifier(
            modality="vision",
            vlm_model_id="mock-model",
            seed=42
        )
        
        mock_vlm = Mock()
        mock_vlm.generate_response = Mock(return_value="Vision response")
        vision_classifier.vlm_wrapper = mock_vlm
        vision_classifier.logger = Mock()
        vision_classifier.class_names = ["Cat", "Dog"]
        vision_classifier.effective_model_id = "mock-vision-model"
        
        vision_classifier._last_prediction_context = {
            'task_type': 'classification',
            'class_names': ["Cat", "Dog"],
            'num_test_samples': 10,
            'completed_samples': 10,
            'completion_rate': 1.0,
            'modality': 'vision',
            'recent_predictions': [],
            'timestamp': time.time()
        }
        
        vision_classifier.chat("What did you see in the images?")
        
        call_args = vision_classifier.vlm_wrapper.generate_response.call_args
        prompt = call_args[1]['text_input']
        
        assert "Modality: vision" in prompt
        assert "vision data" in prompt

    def test_get_chat_history_structure(self):
        """Test the structure of returned chat history."""
        classifier = self.create_mock_classifier()
        
        # Add some chat history
        classifier.chat("First question")
        time.sleep(0.01)  # Ensure different timestamps
        classifier.chat("Second question")
        
        history = classifier.get_chat_history()
        
        assert isinstance(history, list)
        assert len(history) == 2
        
        for exchange in history:
            assert isinstance(exchange, dict)
            assert 'user' in exchange
            assert 'assistant' in exchange
            assert 'timestamp' in exchange
            assert isinstance(exchange['timestamp'], float)

    def test_chat_integration_with_predict(self):
        """Test chat integration - simplified to focus on context creation."""
        classifier = self.create_mock_classifier()
        
        # Manually call predict method with mocked prediction context creation
        # This simulates what would happen after a real predict() call
        classifier._last_prediction_context = {
            'task_type': 'classification',
            'class_names': self.class_names,
            'num_test_samples': 3,
            'completed_samples': 3,
            'completion_rate': 1.0,
            'modality': 'tabular',
            'recent_predictions': [
                {'predicted_class': 'Class A', 'true_class': 'Class A', 'confidence': 'N/A'},
                {'predicted_class': 'Class B', 'true_class': 'Class C', 'confidence': 'N/A'}
            ],
            'timestamp': time.time()
        }
        
        # Now test chat - should have prediction context
        response = classifier.chat("What patterns did you observe?")
        
        assert isinstance(response, str)
        assert response == "This is a mock chat response."
        
        # Verify prediction context exists and is used
        assert hasattr(classifier, '_last_prediction_context')
        assert classifier._last_prediction_context is not None
        assert classifier._last_prediction_context['modality'] == 'tabular'
        assert classifier._last_prediction_context['num_test_samples'] == 3

    @pytest.mark.parametrize("modality", ["tabular", "vision", "audio"])
    def test_chat_works_with_all_modalities(self, modality):
        """Test that chat works with all supported modalities."""
        classifier = self.create_mock_classifier()
        classifier.modality = modality
        classifier._last_prediction_context['modality'] = modality
        
        response = classifier.chat(f"Tell me about the {modality} analysis")
        
        assert isinstance(response, str)
        assert response == "This is a mock chat response."

    def test_chat_response_parsing_different_formats(self):
        """Test parsing different VLM response formats."""
        classifier = self.create_mock_classifier()
        
        # Test dict response with 'text' key
        classifier.vlm_wrapper.generate_response.return_value = {"text": "Dict text response"}
        response1 = classifier.chat("Question 1")
        assert response1 == "Dict text response"
        
        # Test dict response with 'response' key
        classifier.vlm_wrapper.generate_response.return_value = {"response": "Dict response response"}
        response2 = classifier.chat("Question 2")
        assert response2 == "Dict response response"
        
        # Test string response
        classifier.vlm_wrapper.generate_response.return_value = "String response"
        response3 = classifier.chat("Question 3")
        assert response3 == "String response"




if __name__ == "__main__":
    pytest.main([__file__, "-v"])