import torch
from torch import nn
from typing import Iterator, Any, List, Sequence, Tuple, Union
import logging
from tools import config as gemma_config
from tools.model import GemmaDecoderLayer, RMSNorm
from tools.network_utils import receive_data, transmit_data_to_next_nodes
import socket
import signal
import time
import random
import streamlit as st

class GemmaFirstLayerModel(nn.Module):
    """
    Model for the first layer of the Gemma model. Uses a single GemmaDecoderLayer.
    
    Args:
        config (gemma_config.GemmaConfig): Configuration object for the model.
    """
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.first_layer = GemmaDecoderLayer(config)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the first layer of the model.
        
        Args:
            hidden_states (torch.Tensor): The hidden states of the input.
            freqs_cis (torch.Tensor): Rotary positional embeddings.
            kv_write_indices (torch.Tensor): Indices for key-value caches.
            kv_cache (Tuple[torch.Tensor, torch.Tensor]): Key-value cache tuple.
            mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Updated hidden states.
        """
        hidden_states = self.first_layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_cache, mask=mask)
        return hidden_states

class GemmaRemainingLayersModel(nn.Module):
    """
    Model for the remaining layers of the Gemma model, excluding the first layer.
    Includes normalization after the layers.

    Args:
        config (gemma_config.GemmaConfig): Configuration object for the model.
    """
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(1, config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the remaining layers of the model.
        
        Args:
            hidden_states (torch.Tensor): The hidden states of the input.
            freqs_cis (torch.Tensor): Rotary positional embeddings.
            kv_write_indices (torch.Tensor): Indices for key-value caches.
            kv_caches (List[Tuple[torch.Tensor, torch.Tensor]]): List of key-value caches for each layer.
            mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Updated hidden states after all layers and normalization.
        """
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_caches[i], mask=mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class GemmaLayerModel(nn.Module):
    """
    Model for a single layer of the Gemma model.
    
    Args:
        config (gemma_config.GemmaConfig): Configuration object for the model.
    """
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layer = GemmaDecoderLayer(config)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for a single layer of the model.
        
        Args:
            hidden_states (torch.Tensor): The hidden states of the input.
            freqs_cis (torch.Tensor): Rotary positional embeddings.
            kv_write_indices (torch.Tensor): Indices for key-value caches.
            kv_cache (Tuple[torch.Tensor, torch.Tensor]): Key-value cache tuple.
            mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Updated hidden states.
        """
        hidden_states = self.layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_cache, mask=mask)
        return hidden_states
    
class GemmaLastLayerModel(nn.Module):
    """
    Model for the last layer of the Gemma model, with an additional normalization step.
    
    Args:
        config (gemma_config.GemmaConfig): Configuration object for the model.
    """
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layer = GemmaDecoderLayer(config)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_cache: Tuple[torch.Tensor, torch.Tensor], mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the last layer of the model.
        
        Args:
            hidden_states (torch.Tensor): The hidden states of the input.
            freqs_cis (torch.Tensor): Rotary positional embeddings.
            kv_write_indices (torch.Tensor): Indices for key-value caches.
            kv_cache (Tuple[torch.Tensor, torch.Tensor]): Key-value cache tuple.
            mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Updated hidden states after the last layer and normalization.
        """
        hidden_states = self.layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_cache, mask=mask)
        hidden_states = self.norm(hidden_states)
        return hidden_states
    
class GemmaMiddleLayersModel(nn.Module):
    """
    Model for the middle layers of the Gemma model, excluding the first and last layers.
    
    Args:
        config (gemma_config.GemmaConfig): Configuration object for the model.
    """
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([GemmaDecoderLayer(config) for _ in range(1, config.num_hidden_layers - 1)])
        
    def forward(self, hidden_states: torch.Tensor, freqs_cis: torch.Tensor, kv_write_indices: torch.Tensor, kv_caches: List[Tuple[torch.Tensor, torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the middle layers of the model.
        
        Args:
            hidden_states (torch.Tensor): The hidden states of the input.
            freqs_cis (torch.Tensor): Rotary positional embeddings.
            kv_write_indices (torch.Tensor): Indices for key-value caches.
            kv_caches (List[Tuple[torch.Tensor, torch.Tensor]]): List of key-value caches for each layer.
            mask (torch.Tensor): Attention mask for the input.

        Returns:
            torch.Tensor: Updated hidden states after the middle layers.
        """
        for i, layer in enumerate(self.layers):
            hidden_states = layer(hidden_states=hidden_states, freqs_cis=freqs_cis, kv_write_indices=kv_write_indices, kv_cache=kv_caches[i], mask=mask)
        return hidden_states

def load_model(model, filepath):
    """
    Loads a model's state dictionary from a specified file path.

    Args:
        model (torch.nn.Module): The model object to load the weights into.
        filepath (str): Path to the file containing the model's state dictionary.

    Raises:
        Exception: If there is an issue with loading the model.
    """
    try:
        model.load_state_dict(torch.load(filepath, weights_only=True))
        logging.info(f"Model loaded successfully from {filepath}")
    except Exception as e:
        logging.error(f"Failed to load model from {filepath}: {e}")
        raise

def forward_pass(
    next_layers: dict,
    server_socket: socket.socket,
    max_retries: int,
    reconnection_freq: int,
    config: Any,
    model: List[nn.Module],
    embedder: nn.Module,
    sampler: nn.Module,
    prec_freqs_cis: torch.Tensor,
    output_index: torch.Tensor,
    device: torch.device,
    input_token_ids: torch.Tensor,
    input_positions: torch.Tensor,
    kv_write_indices: torch.Tensor,
    kv_caches: List[Tuple[torch.Tensor, torch.Tensor]],
    mask: torch.Tensor,
    output_positions: torch.Tensor,
    temperatures: Union[torch.Tensor, None],
    top_ps: torch.Tensor,
    top_ks: torch.Tensor,
    **kwargs,
) -> torch.Tensor:
    """
    Executes a forward pass for the model and communicates with the next layer.

    Args:
        next_layers (dict): Dictionary containing information about the next layers.
        server_socket (socket.socket): The server socket for handling client connections.
        max_retries (int): Maximum number of retries if forward pass fails to reach starting node.
        reconnection_freq (int): Number of tokens before trying to reconnect to unavailable nodes.
        config (Any): Configuration for the model.
        model (List[torch.nn.Module]): List of model layers.
        embedder (torch.nn.Module): The embedding layer of the model.
        sampler (torch.nn.Module): The sampler for generating tokens.
        prec_freqs_cis (torch.Tensor): Precomputed frequencies for rotary embeddings.
        output_index (torch.Tensor): Output index tensor for token generation.
        device (torch.device): Device on which to run computations.
        input_token_ids (torch.Tensor): Input token IDs tensor.
        input_positions (torch.Tensor): Input positions tensor.
        kv_write_indices (torch.Tensor): Key-value write indices.
        kv_caches (List[Tuple[torch.Tensor, torch.Tensor]]): List of key-value caches.
        mask (torch.Tensor): Mask tensor for attention.
        output_positions (torch.Tensor): Output positions tensor for sampling.
        temperatures (Union[torch.Tensor, None]): Temperature tensor for sampling.
        top_ps (torch.Tensor): Tensor for top-p sampling.
        top_ks (torch.Tensor): Tensor for top-k sampling.

    Returns:
        torch.Tensor: The generated next tokens.
    """
    freqs_cis = prec_freqs_cis.index_select(0, input_positions)
    kv_write_indices = input_positions

    # [batch_size, input_len, hidden_size]
    hidden_states = embedder(input_token_ids)
    hidden_states = hidden_states * (config.hidden_size**0.5)
    hidden_states = model[0](
        hidden_states=hidden_states,
        freqs_cis=freqs_cis,
        kv_write_indices=kv_write_indices,
        kv_cache=kv_caches[0],
        mask=mask,
    )
    # Construct dictionary containing all the data
    data = {
        'hidden_state': hidden_states,
        'KV_index': kv_write_indices,
    }
    # Set up the socket
    logging.debug("Preparing to send data: {}".format(data))
    try_to_reconnect = output_index % reconnection_freq == 0
    exit_status = transmit_data_to_next_nodes(data, next_layers, try_to_reconnect)

    # Try to receive response from last layer with retry mechanism
    retry_count = 0
    deserialized_response = None
    while retry_count < max_retries:
        try:
            # Attempt to accept a connection from the last layer
            client_socket, client_addr = server_socket.accept()
            logging.debug(f"Accepted connection from {client_addr[0]}:{client_addr[1]}")
            deserialized_response = receive_data(client_socket)
            client_socket.close()
            # Successfully received response, break out of retry loop
            break
        except socket.timeout:
            # Timed out waiting for a response
            logging.warning(f"Timeout waiting for last layer response (attempt {retry_count + 1}/{max_retries}). Retrying...")
            # Resend the data to next nodes before retrying accept
            exit_status = transmit_data_to_next_nodes(data, next_layers, try_to_reconnect)
            if exit_status != 0:
                logging.error("Failed to resend data after timeout. Will retry accepting again.")
            retry_count += 1
        except Exception as e:
            logging.error(f"Error while waiting for last layer response: {e}")
            break

    if deserialized_response is None:
        # Failed to receive response after all retries
        logging.error("Failed to receive a response from the next layers after multiple retries. Aborting inference.")
        # You can choose to return or raise an exception here
        raise Exception("Inference process failed: User never received a response.")
    
    # Extract tensor data from response
    hidden_states = deserialized_response['hidden_state']

    embedder_weight = embedder.weight
    if config.quant:
        embedder_weight = (
            embedder_weight * embedder.weight_scaler.unsqueeze(-1))
    next_tokens = sampler(
        embedding=embedder_weight,
        hidden_states=hidden_states.to(device), # After coming back from the server
        output_positions=output_positions,
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
    )
    return next_tokens
        
def generation(
    layer_info: dict,
    next_layers: List[dict],
    tokenizer: Any,
    config: Any,
    model: List[torch.nn.Module],
    embedder: torch.nn.Module,
    sampler: torch.nn.Module,
    prec_freqs_cis: torch.Tensor,
    prompts: Union[str, Sequence[str]],
    device: Any,
    output_len: int = 100,
    temperature: Union[float, None] = 0.95,
    top_p: float = 1.0,
    top_k: int = 100,
    time_out: int = 40,
    max_retries: int = 4,
    reconnection_freq: int = 10, 
) -> Union[str, Sequence[str]]:
    """
    Generates responses for given prompts using the Gemma model.

    Args:
        layer_info (dict): Information about the current layer.
        next_layers (List[dict]): Information about the next layers.
        tokenizer (Any): Tokenizer used for encoding and decoding.
        config (Any): Configuration for the model.
        model (List[torch.nn.Module]): List of model layers.
        embedder (torch.nn.Module): Embedding layer of the model.
        sampler (torch.nn.Module): Sampler for generating tokens.
        prec_freqs_cis (torch.Tensor): Precomputed frequencies for rotary embeddings.
        prompts (Union[str, Sequence[str]]): Input prompt(s) for generation.
        device (Any): Device to run computations on.
        output_len (int): Length of the output sequence.
        temperature (Union[float, None]): Temperature value for sampling.
        top_p (float): Top-p sampling parameter.
        top_k (int): Top-k sampling parameter.
        time_out (int): Forward pass timeout to resend weights in case of no response.
        max_retries (int): Maximum number of retries if forward pass fails to reach starting node.
        reconnection_freq (int): Number of tokens before trying to reconnect to unavailable nodes.

    Returns:
        Union[str, Sequence[str]]: Generated responses.
    """
    try:
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
        logging.debug(f"Length of input prompt: {len(prompt_tokens[0])}")
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= config.max_position_embeddings

        # build KV caches
        kv_caches = []
        for i in range(len(model)):
            size = (batch_size, max_seq_len, config.num_key_value_heads,
                    config.head_dim)
            dtype = config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                    tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            tokenizer.pad_id,
                                            dtype=torch.int64)
        
        # Necessary data to instantiate structures at the server side
        data = {'batch_size': batch_size,
                'max_seq_len': max_seq_len}
        # Send data to next nodes
        exit_status = transmit_data_to_next_nodes(data, next_layers)

        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                            dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.

        # Initialize server socket to listen for weights coming from last layer
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((layer_info['host'], layer_info['port']))
        server_socket.listen(5)
        server_socket.settimeout(time_out)
        logging.info(f"Server listening on {layer_info['host']}:{layer_info['port']}...")

        for i in range(max_seq_len - min_prompt_len):
            logging.debug("Forward pass {} start: {:.6f}\n".format(i+1, time.time()))
            current_start = time.perf_counter()
            next_token_ids = forward_pass(
                next_layers,
                server_socket,
                max_retries,
                reconnection_freq,
                config,
                model,
                embedder,
                sampler,
                prec_freqs_cis,
                output_index,
                device,
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
            )
            logging.debug("Forward pass {} successful".format(i+1))
            current_end = time.perf_counter()
            execution_time = current_end - current_start
            logging.debug(f"Execution time: {execution_time:.6f} seconds")
            logging.info("Forward pass {} execution time: {:.6f}\n".format(i+1, execution_time))
            
            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                        next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
            output_index = output_index + 1
    finally:
        server_socket.close()
    # Detokenization.
    token_ids = token_ids_tensor.tolist()
    results = []
    for i, tokens in enumerate(token_ids):
        trimmed_output = tokens[len(prompt_tokens[i]):len(prompt_tokens[i])
                                + output_len]
        if tokenizer.eos_id in trimmed_output:
            eos_index = trimmed_output.index(tokenizer.eos_id)
            trimmed_output = trimmed_output[:eos_index]
        results.append(tokenizer.decode(trimmed_output))

    # If a string was provided as input, return a string as output.
    return results[0] if is_str_prompt else results

def stream_generation(
    layer_info: dict,
    next_layers: dict,
    tokenizer: Any,
    config: Any,
    model: List[torch.nn.Module],
    embedder: torch.nn.Module,
    sampler: torch.nn.Module,
    prec_freqs_cis: torch.Tensor,
    prompts: Union[str, Sequence[str]],
    device: Any,
    output_len: int = 100,
    temperature: Union[float, None] = 0.95,
    top_p: float = 1.0,
    top_k: int = 100,
    time_out: int = 40,
    max_retries: int = 4,
    reconnection_freq: int = 10,
) -> Iterator[Union[str, Sequence[str]]]:
    """
    Generates responses for given prompts using Gemma model in a streaming manner.

    Args:
        layer_info (dict): Information about the current layer.
        next_layers (dict): Information about the next layers.
        tokenizer (Any): Tokenizer used for encoding and decoding.
        config (Any): Configuration for the model.
        model (List[torch.nn.Module]): List of model layers.
        embedder (torch.nn.Module): Embedding layer of the model.
        sampler (torch.nn.Module): Sampler for generating tokens.
        prec_freqs_cis (torch.Tensor): Precomputed frequencies for rotary embeddings.
        prompts (Union[str, Sequence[str]]): Input prompt(s) for generation.
        device (Any): Device to run computations on.
        output_len (int): Length of the output sequence.
        temperature (Union[float, None]): Temperature value for sampling.
        top_p (float): Top-p sampling parameter.
        top_k (int): Top-k sampling parameter.
        time_out (int): Forward pass timeout to resend weights in case of no response.
        max_retries (int): Maximum number of retries if forward pass fails to reach starting node.
        reconnection_freq (int): Number of tokens before trying to reconnect to unavailable nodes.

    Yields:
        Union[str, Sequence[str]]: Generated responses in a streaming fashion.
    """

    try:
        # If a single prompt is provided, treat it as a batch of 1.
        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens = [tokenizer.encode(prompt) for prompt in prompts]
        logging.debug(f"Length of input prompt: {len(prompt_tokens[0])}")
        min_prompt_len = min(len(p) for p in prompt_tokens)
        max_prompt_len = max(len(p) for p in prompt_tokens)
        max_seq_len = max_prompt_len + output_len
        assert max_seq_len <= config.max_position_embeddings, "Sequence length exceeds model capacity."

        # build KV caches
        kv_caches = []
        for i in range(len(model)):
            size = (batch_size, max_seq_len, config.num_key_value_heads,
                    config.head_dim)
            dtype = config.get_dtype()
            k_cache = torch.zeros(size=size, dtype=dtype, device=device)
            v_cache = torch.zeros(size=size, dtype=dtype, device=device)
            kv_caches.append((k_cache, v_cache))

        # prepare inputs
        token_ids_tensor = torch.full((batch_size, max_seq_len),
                                    tokenizer.pad_id, dtype=torch.int64)
        input_token_ids_tensor = torch.full((batch_size, min_prompt_len),
                                            tokenizer.pad_id,
                                            dtype=torch.int64)

        # Necessary data to instantiate structures at the server sides
        data = {'batch_size': batch_size,
                'max_seq_len': max_seq_len}
        # Send data to next nodes
        exit_status = transmit_data_to_next_nodes(data, next_layers)

        for i, p in enumerate(prompt_tokens):
            token_ids_tensor[i, :len(p)] = torch.tensor(p)
            input_token_ids_tensor[i, :min_prompt_len] = torch.tensor(
                p[:min_prompt_len])
        token_ids_tensor = token_ids_tensor.to(device)
        input_token_ids_tensor = input_token_ids_tensor.to(device)
        prompt_mask_tensor = token_ids_tensor != tokenizer.pad_id
        input_positions_tensor = torch.arange(0, min_prompt_len,
                                            dtype=torch.int64).to(device)
        mask_tensor = torch.full((1, 1, max_seq_len, max_seq_len),
                                -2.3819763e38).to(torch.float)
        mask_tensor = torch.triu(mask_tensor, diagonal=1).to(device)
        curr_mask_tensor = mask_tensor.index_select(2, input_positions_tensor)
        output_positions_tensor = torch.LongTensor([min_prompt_len - 1]).to(
            device)
        temperatures_tensor = None if not temperature else torch.FloatTensor(
            [temperature] * batch_size).to(device)
        top_ps_tensor = torch.FloatTensor([top_p] * batch_size).to(device)
        top_ks_tensor = torch.LongTensor([top_k] * batch_size).to(device)
        output_index = torch.tensor(min_prompt_len, dtype=torch.int64).to(
            device)

        # Prefill up to min_prompt_len tokens, then treat other prefill as
        # decode and ignore output.

        # Initialize server socket to listen for weights coming from last layer
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((layer_info['host'], layer_info['port']))
        server_socket.listen(5)
        server_socket.settimeout(time_out)
        logging.info(f"Server listening on {layer_info['host']}:{layer_info['port']}...")

        # Initialize a tensor to track which sequences have generated EOS.
        done = torch.zeros(batch_size, dtype=torch.bool, device=device)
        eos_token_id = tokenizer.eos_id

        for i in range(max_seq_len - min_prompt_len):
            logging.info("Forward pass {} start: {:.6f}\n".format(i+1, time.time()))
            current_start = time.perf_counter()
            next_token_ids = forward_pass(
                next_layers,
                server_socket,
                max_retries,
                reconnection_freq,
                config,
                model,
                embedder,
                sampler,
                prec_freqs_cis,
                output_index,
                device,
                input_token_ids=input_token_ids_tensor,
                input_positions=input_positions_tensor,
                kv_write_indices=None,
                kv_caches=kv_caches,
                mask=curr_mask_tensor,
                output_positions=output_positions_tensor,
                temperatures=temperatures_tensor,
                top_ps=top_ps_tensor,
                top_ks=top_ks_tensor,
            )
            logging.debug("Forward pass {} successful".format(i+1))
            current_end = time.perf_counter()
            execution_time = current_end - current_start
            logging.debug(f"Execution time: {execution_time:.6f} seconds")
            logging.info("Forward pass {} execution time: {:.6f}\n".format(i+1, execution_time))
            
            curr_prompt_mask = prompt_mask_tensor.index_select(
                1, output_index).squeeze(dim=1)
            curr_token_ids = token_ids_tensor.index_select(
                1, output_index).squeeze(dim=1)
            output_token_ids = torch.where(curr_prompt_mask, curr_token_ids,
                                        next_token_ids).unsqueeze(dim=1)
            token_ids_tensor.index_copy_(1, output_index, output_token_ids)

            # Extract the generated token(s)
            generated_token_ids = next_token_ids.tolist()
            generated_tokens = [tokenizer.decode([token_id]) for token_id in generated_token_ids]
            if is_str_prompt:
                logging.debug(f"Returning word {generated_tokens[0]}")
                yield generated_tokens[0]
            else:
                yield generated_tokens  # For batch processing, yields a list of tokens

            # Update the 'done' tensor to track which sequences have generated EOS.
            # done |= (next_token_ids == eos_token_id)

            # If all sequences have generated EOS, break the loop.
            if done.all():
                logging.info("All sequences have generated EOS. Stopping generation.")
                break

            input_token_ids_tensor = output_token_ids
            input_positions_tensor = output_index.unsqueeze(dim=-1)
            curr_mask_tensor = mask_tensor.index_select(2,
                                                        input_positions_tensor)
            output_positions_tensor = torch.tensor(0, dtype=torch.int64).to(
                device)
            output_index = output_index + 1
    finally:
        server_socket.close()


# Streamed response emulator
def response_generator(*args):
    """
    Simulates a random chatbot response.
    
    Returns:
        str: A random response message.
    """
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )
    logging.debug(f"Generated response: {response}")
    return response

def dummy_stream(*args, **kwargs):
    """
    Dummy generator for simulating a streamed response.
    """
    for i in range(5):
        yield "a "
        time.sleep(2)

def clear_chat():
    """
    Clears the chat history in the Streamlit session.
    """
    st.session_state.chat_history = ""
    st.session_state.messages = []
    logging.info("Chat history cleared")