import streamlit as st
import os
import torch
import numpy as np
import argparse
from tools.config import get_config_for_7b, get_config_for_2b
from tools.model import Sampler, Embedding, precompute_freqs_cis
from tools.tokenizer import Tokenizer
from tools.model_utils import GemmaLayerModel, GemmaLastLayerModel, load_model, stream_generation, generation, clear_chat, response_generator, dummy_stream
import time
import logging
import sys

def setup_logging(layer):
    """
    Set up logging for the specified layer.

    Args:
        layer (int): The index of the layer for which logging is initialized.
    """
    log_dir = './logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'layer_{layer}.log')
    logging.basicConfig(
        level=logging.ERROR,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Logging initialized for layer {layer}")

def initialize_model(next_layers):
    """
    Initialize the model, tokenizer, embedder, and sampler. Load pre-trained weights and configure the model layers.

    Args:
        next_layers (list): List of dictionaries containing host and port information of the next layers.
    """

    # Model variant and machine type configuration
    VARIANT = '1.1-2b-it'
    MACHINE_TYPE = 'cuda'

    # Select configuration based on model variant
    config = get_config_for_2b() if "2b" in VARIANT else get_config_for_7b()
    config.tokenizer = f'./weights/{VARIANT}/tokenizer.model'

    # Assert that we have recieved enough number of next hops
    total_layers = config.num_hidden_layers
    assert len(next_layers) == 5, "UE expects the information (ip:port) of the next three layers"

    # Ensure tokenizer model file exists
    if not os.path.isfile(config.tokenizer):
        logging.error("Tokenizer not found!")
        sys.exit(1)

    # Set quantization and device type
    config.quant = 'quant' in VARIANT
    torch.set_default_dtype(config.get_dtype())
    st.session_state.device = torch.device('cuda' if torch.cuda.is_available() and MACHINE_TYPE == 'cuda' else 'cpu')

    # Initialize and load the first model layer
    first_layer_model = GemmaLayerModel(config) 
    load_model(first_layer_model, f'./weights/{VARIANT}/layer_model_0.pth')
    first_layer_model.to(st.session_state.device)
    first_layer_model.eval()
    logging.info("First layer model initialized and loaded.")

    # Initialize the last layer model if needed
    st.session_state.model = [first_layer_model]
    

    # Initialize tokenizer, embedding, and sampler
    st.session_state.tokenizer = Tokenizer(config.tokenizer)
    embedder = Embedding(config.vocab_size, config.hidden_size, config.quant)
    load_model(embedder, f'./weights/{VARIANT}/embedding_weights.pth')
    embedder.to(st.session_state.device)
    embedder.eval()
    st.session_state.embedder = embedder
    st.session_state.sampler = Sampler(config.vocab_size).to(st.session_state.device)

    # Pre-compute rotary embedding table (frequencies for positional encoding)
    rope_theta = getattr(config, 'rope_theta', 10000)
    st.session_state.prec_freqs_cis = precompute_freqs_cis(config.head_dim, config.max_position_embeddings * 2, theta=rope_theta).to(st.session_state.device)
    st.session_state.config = config

# Argument parsing (for possible backend adjustments)
def parse_args():
    """
    Parse command-line arguments for configuring host and port settings.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Configure host and port for remote node communication.")
    parser.add_argument("--host", type=str, default='127.0.0.1', help="Host IP from current layer")
    parser.add_argument("--port", type=int, default=12647, help="Port from current layer")
    parser.add_argument("--next_layers", type=str, nargs='+', required=True, help="Next 3 layers' host:port")
    return parser.parse_args()

# Main application entry point
if __name__ == "__main__":
    if "initialized" not in st.session_state or not st.session_state.initalized:
        setup_logging(layer=0)
        # Parse arguments
        args = parse_args()
        logging.debug(f"This code is being run with Numpy: {str(np.__version__)}")
        
        # Host and port configurations for previous and next layers
        st.session_state.layer_info = {'host': args.host, 'port': args.port}
        st.session_state.next_layers = [{'host': layer.split(':')[0], 'port': layer.split(':')[1], 
                        'layer': layer_idx + 1, 'available': True} 
                    for layer_idx, layer in enumerate(args.next_layers)]
        
        initialize_model(st.session_state.next_layers)

    # Chat templates
    USER_CHAT_TEMPLATE = '<start_of_turn>user\n{prompt}<end_of_turn>\n'
    MODEL_CHAT_TEMPLATE = '<start_of_turn>model\n{prompt}<end_of_turn>\n'
    # Streamlit GUI setup
    st.set_page_config(page_title="JARVIS", page_icon="🤖")
    st.title("JARVIS Graphical User Interface")

    # Initialize chat session history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = ""

    # Display chat history on the interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input handling
    if prompt := st.chat_input("Enter your message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Update chat history with user message
        st.session_state.chat_history += USER_CHAT_TEMPLATE.format(prompt=prompt)

        # Prepare for model generation
        with st.chat_message("assistant"):
            assistant_message_placeholder = st.empty()
            assistant_message = ""

            # Generate model response iteratively
            model_response_generator = stream_generation(
                st.session_state.layer_info, st.session_state.next_layers, st.session_state.tokenizer, st.session_state.config, 
                st.session_state.model, st.session_state.embedder, st.session_state.sampler, st.session_state.prec_freqs_cis, 
                prompts=USER_CHAT_TEMPLATE.format(prompt = st.session_state.chat_history + '<start_of_turn>model\n'), device=st.session_state.device, output_len=30
            )

            for token in model_response_generator:
                assistant_message += token
                assistant_message_placeholder.markdown(assistant_message)

            logging.debug(f"Message history:\n{st.session_state.chat_history + assistant_message}")
            logging.info(f"Executed forward pass")

            # Update chat history with model's response
            st.session_state.chat_history += MODEL_CHAT_TEMPLATE.format(prompt=assistant_message)
            st.session_state.messages.append({"role": "assistant", "content": assistant_message})

    # Erase chat history on button press
    if st.button(label="Erase history"):
        clear_chat()
        st.rerun()
