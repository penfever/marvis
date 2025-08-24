#!/usr/bin/env python
"""
Integration tests for MARVIS chat functionality.

These tests verify the chat feature works end-to-end with real MARVIS components,
including prediction workflows, context management, and multi-modal support.
"""

import pytest
import numpy as np
import tempfile
import shutil
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import MARVIS modules
from marvis.models.marvis_tsne import MarvisTsneClassifier


@pytest.mark.integration
class TestMarvisChatIntegration:
    """Integration tests for MARVIS chat functionality."""

    def setup_method(self):
        """Set up integration test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp(prefix='marvis_chat_integration_'))
        
        # Create test datasets for different modalities
        np.random.seed(42)
        
        # Tabular data
        self.X_train_tab = np.random.randn(40, 8)
        self.y_train_tab = np.random.randint(0, 3, 40)
        self.X_test_tab = np.random.randn(15, 8)
        self.y_test_tab = np.random.randint(0, 3, 15)
        self.class_names_tab = ["Alpha", "Beta", "Gamma"]
        
        # Vision data (simulated with smaller feature vectors)
        self.X_train_vis = np.random.rand(30, 64)  # Simulating 8x8 images
        self.y_train_vis = np.random.randint(0, 2, 30)
        self.X_test_vis = np.random.rand(10, 64)
        self.y_test_vis = np.random.randint(0, 2, 10)
        self.class_names_vis = ["Circle", "Square"]
        
    def teardown_method(self):
        """Clean up integration test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def create_mock_vlm_responses(self, response_type="tabular"):
        """Create appropriate mock VLM responses for different scenarios."""
        if response_type == "tabular":
            return [
                "Based on the tabular data analysis, the model achieved 80% accuracy on 15 test samples. The t-SNE visualization revealed three distinct clusters corresponding to Alpha, Beta, and Gamma classes, with good separation indicating the TabPFN embeddings captured meaningful patterns.",
                "The visualization shows clear clustering patterns where similar data points group together. The Alpha class forms a tight cluster in the upper region, Beta samples cluster in the lower left, and Gamma samples are concentrated in the right portion of the plot.",
                "To improve classification accuracy, I recommend: 1) Increasing the training set size for better embeddings, 2) Experimenting with different t-SNE perplexity values (currently optimal), 3) Considering 3D visualization for complex patterns, and 4) Using cross-validation to validate model stability.",
                "The model shows high confidence in its predictions, with clear decision boundaries visible in the t-SNE space. The KNN connections (k=5) confirm that similar samples are properly grouped, suggesting reliable classification performance.",
                "The misclassified samples appear near cluster boundaries, which is expected behavior. These boundary cases often represent genuine ambiguity in the data rather than model errors."
            ]
        elif response_type == "vision":
            return [
                "The vision model achieved 85% accuracy on image classification using DINOV2 embeddings. The t-SNE visualization shows clear separation between Circle and Square classes, indicating that visual features were effectively captured.",
                "In the visual embedding space, I observe two distinct clusters representing the geometric shapes. Circle samples form a cohesive group due to their curved features, while Square samples cluster based on their angular characteristics.",
                "For vision improvement, consider: 1) Augmenting training data with rotations and scaling, 2) Using higher resolution embeddings if computational resources allow, 3) Experimenting with different vision encoders like BioCLIP2 for specialized domains."
            ]
        elif response_type == "error_handling":
            return [
                "I encountered an issue processing this request. Let me provide general guidance based on typical MARVIS workflows.",
                "Based on standard classification patterns, accuracy can often be improved through better data preprocessing and parameter tuning."
            ]
        else:
            return ["This is a generic response for testing purposes."]

    @pytest.mark.integration
    def test_tabular_chat_end_to_end(self):
        """Test complete chat workflow with tabular data."""
        # Create classifier with fast settings
        classifier = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id="mock-tabular-model",
            tsne_perplexity=8,
            tsne_max_iter=50,
            nn_k=3,
            seed=42
        )
        
        # Mock VLM with appropriate responses
        mock_vlm = Mock()
        mock_responses = self.create_mock_vlm_responses("tabular")
        mock_vlm.generate_response = Mock(side_effect=mock_responses)
        classifier.vlm_wrapper = mock_vlm
        classifier.logger = Mock()
        
        print("\n🧪 Testing Tabular Chat Integration...")
        
        # Set up prediction context manually (simulating successful evaluation)
        # This avoids complex mocking of the entire prediction pipeline
        classifier._last_prediction_context = {
            'task_type': 'classification',
            'class_names': self.class_names_tab,
            'num_test_samples': 5,
            'completed_samples': 5,
            'completion_rate': 1.0,
            'modality': 'tabular',
            'accuracy': 0.80,
            'balanced_accuracy': 0.75,
            'f1_macro': 0.78,
            'evaluation_completed': True,
            'recent_predictions': [
                {'predicted_class': 'Alpha', 'true_class': 'Alpha', 'confidence': 'N/A'},
                {'predicted_class': 'Beta', 'true_class': 'Gamma', 'confidence': 'N/A'},
                {'predicted_class': 'Gamma', 'true_class': 'Gamma', 'confidence': 'N/A'}
            ],
            'timestamp': time.time()
        }
        
        # Test comprehensive chat workflow
        # Chat 1: Performance question
        response1 = classifier.chat("How well did the model perform on the test data?")
        assert isinstance(response1, str)
        assert response1 == mock_responses[0]
        
        # Chat 2: Pattern analysis
        response2 = classifier.chat("What patterns did you observe in the visualization?")
        assert isinstance(response2, str)
        assert response2 == mock_responses[1]
        
        # Chat 3: Improvement suggestions
        response3 = classifier.chat("How could we improve the classification results?")
        assert isinstance(response3, str)
        assert response3 == mock_responses[2]
        
        # Chat 4: Confidence assessment
        response4 = classifier.chat("How confident is the model in its predictions?")
        assert isinstance(response4, str)
        assert response4 == mock_responses[3]
        
        # Chat 5: Error analysis
        response5 = classifier.chat("Which samples were misclassified and why?")
        assert isinstance(response5, str)
        assert response5 == mock_responses[4]
        
        # Verify chat history
        history = classifier.get_chat_history()
        assert len(history) == 5
        assert all('user' in exchange for exchange in history)
        assert all('assistant' in exchange for exchange in history)
        assert all('timestamp' in exchange for exchange in history)
        
        # Verify context was properly included in VLM calls
        assert mock_vlm.generate_response.call_count == 5
        for call_args in mock_vlm.generate_response.call_args_list:
            prompt = call_args[1]['text_input']
            assert "MARVIS Classification Context" in prompt
            assert "tabular" in prompt
            assert "Alpha" in prompt or "Beta" in prompt or "Gamma" in prompt
        
        # Test history management
        classifier.clear_chat_history()
        assert len(classifier.get_chat_history()) == 0
        
        print("✅ Tabular chat integration test passed")

    @pytest.mark.integration
    def test_vision_chat_workflow(self):
        """Test chat functionality with vision data."""
        classifier = MarvisTsneClassifier(
            modality="vision",
            vlm_model_id="mock-vision-model",
            tsne_perplexity=5,
            tsne_max_iter=50,
            nn_k=3,
            seed=42
        )
        
        # Mock VLM for vision responses
        mock_vlm = Mock()
        mock_responses = self.create_mock_vlm_responses("vision")
        mock_vlm.generate_response = Mock(side_effect=mock_responses)
        classifier.vlm_wrapper = mock_vlm
        classifier.logger = Mock()
        
        print("\n🧪 Testing Vision Chat Integration...")
        
        # Set up prediction context manually (simulating successful evaluation)
        classifier._last_prediction_context = {
            'task_type': 'classification',
            'class_names': self.class_names_vis,
            'num_test_samples': len(self.X_test_vis),
            'completed_samples': len(self.X_test_vis),
            'completion_rate': 1.0,
            'modality': 'vision',
            'accuracy': 0.85,
            'recent_predictions': [
                {'predicted_class': 'Circle', 'true_class': 'Circle', 'confidence': 'N/A'},
                {'predicted_class': 'Square', 'true_class': 'Circle', 'confidence': 'N/A'}
            ],
            'timestamp': time.time()
        }
        
        # Test vision-specific questions
        response1 = classifier.chat("How did the vision model perform?")
        assert response1 == mock_responses[0]
        
        response2 = classifier.chat("What visual patterns were identified?")
        assert response2 == mock_responses[1]
        
        response3 = classifier.chat("How can I improve visual classification?")
        assert response3 == mock_responses[2]
        
        # Verify vision context in prompts
        for call_args in mock_vlm.generate_response.call_args_list:
            prompt = call_args[1]['text_input']
            assert "vision" in prompt.lower()
            assert "Circle" in prompt or "Square" in prompt
        
        print("✅ Vision chat integration test passed")

    @pytest.mark.integration
    def test_chat_error_handling_and_recovery(self):
        """Test chat error handling and recovery scenarios."""
        classifier = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id="error-prone-model",
            seed=42
        )
        
        # Mock VLM that sometimes fails
        mock_vlm = Mock()
        error_responses = self.create_mock_vlm_responses("error_handling")
        
        # First call succeeds, second fails, third recovers
        mock_vlm.generate_response = Mock(side_effect=[
            "Success response",
            Exception("VLM processing error"),
            error_responses[1]
        ])
        
        classifier.vlm_wrapper = mock_vlm
        classifier.logger = Mock()
        
        # Set up prediction context
        classifier._last_prediction_context = {
            'task_type': 'classification',
            'class_names': self.class_names_tab,
            'num_test_samples': 5,
            'completed_samples': 5,
            'completion_rate': 1.0,
            'modality': 'tabular',
            'timestamp': time.time()
        }
        
        print("\n🧪 Testing Chat Error Handling...")
        
        # First chat - should succeed
        response1 = classifier.chat("This should work")
        assert response1 == "Success response"
        
        # Second chat - should handle error gracefully
        response2 = classifier.chat("This will cause an error")
        assert "I apologize, but I encountered an error" in response2
        assert "VLM processing error" in response2
        
        # Third chat - should recover
        response3 = classifier.chat("This should recover")
        assert response3 == error_responses[1]
        
        # Verify all exchanges were stored in history
        history = classifier.get_chat_history()
        assert len(history) == 3
        assert history[1]['assistant'].startswith("I apologize")  # Error response
        
        print("✅ Chat error handling test passed")

    @pytest.mark.integration  
    def test_chat_context_persistence_across_predictions(self):
        """Test that chat context updates properly across multiple prediction calls."""
        classifier = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id="context-test-model",
            seed=42
        )
        
        mock_vlm = Mock()
        mock_vlm.generate_response = Mock(side_effect=[
            "First evaluation context response",
            "Second evaluation context response", 
            "Combined context response"
        ])
        classifier.vlm_wrapper = mock_vlm
        classifier.logger = Mock()
        
        print("\n🧪 Testing Chat Context Persistence...")
        
        # First prediction context
        classifier._last_prediction_context = {
            'task_type': 'classification',
            'class_names': self.class_names_tab,
            'num_test_samples': 3,
            'completed_samples': 3,
            'completion_rate': 1.0,
            'modality': 'tabular',
            'accuracy': 0.67,
            'timestamp': time.time()
        }
        
        # First chat about initial results
        response1 = classifier.chat("How did the first evaluation go?")
        assert response1 == "First evaluation context response"
        
        # Update context (simulating new prediction)
        classifier._last_prediction_context.update({
            'num_test_samples': 5,
            'completed_samples': 5,
            'accuracy': 0.80,
            'evaluation_completed': True,
            'recent_predictions': [
                {'predicted_class': 'Alpha', 'true_class': 'Alpha', 'confidence': 'N/A'}
            ]
        })
        
        # Second chat with updated context
        response2 = classifier.chat("How about the updated results?")
        assert response2 == "Second evaluation context response"
        
        # Chat referencing conversation history
        response3 = classifier.chat("Compare the first and second evaluations")
        assert response3 == "Combined context response"
        
        # Verify context updates were reflected in prompts
        call_args_list = mock_vlm.generate_response.call_args_list
        
        # First call should mention accuracy 0.67
        first_prompt = call_args_list[0][1]['text_input']
        assert "0.67" in first_prompt
        
        # Second call should mention accuracy 0.80
        second_prompt = call_args_list[1][1]['text_input']
        assert "0.80" in second_prompt or "0.8" in second_prompt
        
        # Third call should have conversation history
        third_prompt = call_args_list[2][1]['text_input']
        assert "Previous Conversation:" in third_prompt
        
        print("✅ Chat context persistence test passed")

    @pytest.mark.integration
    def test_chat_with_different_vlm_interfaces(self):
        """Test chat with different VLM wrapper interface formats."""
        classifier = MarvisTsneClassifier(
            modality="tabular",
            vlm_model_id="interface-test-model",
            seed=42
        )
        
        classifier.logger = Mock()
        classifier._last_prediction_context = {
            'task_type': 'classification',
            'class_names': self.class_names_tab,
            'num_test_samples': 5,
            'completed_samples': 5,
            'completion_rate': 1.0,
            'modality': 'tabular',
            'timestamp': time.time()
        }
        
        print("\n🧪 Testing Different VLM Interface Formats...")
        
        # Test 1: VLM returns dict with 'text' key
        mock_vlm1 = Mock()
        mock_vlm1.generate_response = Mock(return_value={"text": "Dict with text key"})
        classifier.vlm_wrapper = mock_vlm1
        
        response1 = classifier.chat("Test dict text format")
        assert response1 == "Dict with text key"
        
        # Test 2: VLM returns dict with 'response' key
        mock_vlm2 = Mock()
        mock_vlm2.generate_response = Mock(return_value={"response": "Dict with response key"})
        classifier.vlm_wrapper = mock_vlm2
        
        response2 = classifier.chat("Test dict response format") 
        assert response2 == "Dict with response key"
        
        # Test 3: VLM returns plain string
        mock_vlm3 = Mock()
        mock_vlm3.generate_response = Mock(return_value="Plain string response")
        classifier.vlm_wrapper = mock_vlm3
        
        response3 = classifier.chat("Test plain string format")
        assert response3 == "Plain string response"
        
        # Test 4: Fallback to chat method if generate_response not available
        mock_vlm4 = Mock()
        mock_vlm4.chat = Mock(return_value="Chat method response")
        del mock_vlm4.generate_response  # Remove generate_response method
        classifier.vlm_wrapper = mock_vlm4
        
        response4 = classifier.chat("Test chat method fallback")
        assert response4 == "Chat method response"
        
        print("✅ VLM interface format test passed")

    @pytest.mark.integration
    def test_chat_history_limits_and_truncation(self):
        """Test chat history management with limits and truncation."""
        classifier = MarvisTsneClassifier(
            modality="tabular", 
            vlm_model_id="history-test-model",
            seed=42
        )
        
        mock_vlm = Mock()
        # Create enough responses for the test
        mock_responses = [f"Response {i}" for i in range(20)]
        mock_vlm.generate_response = Mock(side_effect=mock_responses)
        classifier.vlm_wrapper = mock_vlm
        classifier.logger = Mock()
        
        classifier._last_prediction_context = {
            'task_type': 'classification',
            'class_names': self.class_names_tab,
            'num_test_samples': 5,
            'completed_samples': 5,
            'completion_rate': 1.0,
            'modality': 'tabular',
            'timestamp': time.time()
        }
        
        print("\n🧪 Testing Chat History Management...")
        
        # Add many chat exchanges
        for i in range(15):
            classifier.chat(f"Question {i}")
        
        # Verify full history exists
        full_history = classifier.get_chat_history()
        assert len(full_history) == 15
        
        # Test limited history in context (max_history=5)
        classifier.chat("Final question with limited history", max_history=5)
        
        # Check that prompt included only recent history
        final_prompt = mock_vlm.generate_response.call_args_list[-1][1]['text_input']
        
        # Should include recent questions (10-14) but not early ones (0-5)
        assert "Question 14" in final_prompt
        assert "Question 13" in final_prompt
        assert "Question 11" in final_prompt  # Within last 5
        assert "Question 5" not in final_prompt  # Should be excluded
        assert "Question 0" not in final_prompt  # Should be excluded
        
        # Test history clearing
        classifier.clear_chat_history()
        assert len(classifier.get_chat_history()) == 0
        
        print("✅ Chat history management test passed")

    @pytest.mark.integration
    def test_comprehensive_multi_modal_chat_workflow(self):
        """Test comprehensive chat workflow across different modalities."""
        modality_configs = [
            ("tabular", self.class_names_tab),
            ("vision", self.class_names_vis),
            ("audio", ["Sound A", "Sound B"])
        ]
        
        print("\n🧪 Testing Multi-Modal Chat Workflow...")
        
        for modality, class_names in modality_configs:
            print(f"   Testing {modality} modality...")
            
            classifier = MarvisTsneClassifier(
                modality=modality,
                vlm_model_id=f"{modality}-test-model",
                seed=42
            )
            
            # Mock VLM with modality-specific responses
            mock_vlm = Mock()
            mock_vlm.generate_response = Mock(return_value=f"This is a {modality} response")
            classifier.vlm_wrapper = mock_vlm
            classifier.logger = Mock()
            
            # Set up modality-specific context
            classifier._last_prediction_context = {
                'task_type': 'classification',
                'class_names': class_names,
                'num_test_samples': 5,
                'completed_samples': 5,
                'completion_rate': 1.0,
                'modality': modality,
                'accuracy': 0.75,
                'timestamp': time.time()
            }
            
            # Test chat
            response = classifier.chat(f"Tell me about the {modality} analysis")
            assert response == f"This is a {modality} response"
            
            # Verify modality-specific context in prompt
            prompt = mock_vlm.generate_response.call_args[1]['text_input']
            assert f"Modality: {modality}" in prompt
            assert f"{modality} data" in prompt
            
        print("✅ Multi-modal chat workflow test passed")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])