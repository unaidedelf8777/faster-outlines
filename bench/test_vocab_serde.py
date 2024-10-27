import time
from transformers import AutoTokenizer
import pickle
import tempfile
import os
from faster_outlines.fsm import TokenVocabulary,  create_token_vocabulary_from_tokenizer


def test_token_vocabulary_serde(tok_name: str):
    # Load tokenizer
    print("Loading tokenizer...")
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(tok_name)
    tokenizer_load_time = time.time() - start
    print(f"Tokenizer loaded in {tokenizer_load_time:.2f}s")

    # Create TokenVocabulary
    print("\nCreating TokenVocabulary...")
    start = time.time()
    vocab = create_token_vocabulary_from_tokenizer(tokenizer)
    vocab_create_time = time.time() - start
    print(f"TokenVocabulary created in {vocab_create_time:.2f}s")
    print(f"Vocabulary size: {vocab.len()} tokens")

    # Serialize
    print("\nSerializing TokenVocabulary...")
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        start = time.time()
        pickle.dump(vocab, tmp)
        serialize_time = time.time() - start
        tmp_path = tmp.name
    
    file_size = os.path.getsize(tmp_path) / (1024 * 1024)  # Convert to MB
    print(f"Serialized in {serialize_time:.2f}s")
    print(f"Serialized size: {file_size:.2f}MB")

    # Deserialize
    print("\nDeserializing TokenVocabulary...")
    start = time.time()
    with open(tmp_path, 'rb') as f:
        deserialized_vocab = pickle.load(f)
    deserialize_time = time.time() - start
    print(f"Deserialized in {deserialize_time:.2f}s")

    # Cleanup
    os.unlink(tmp_path)

    # Verify
    print("\nVerifying deserialization...")
    assert vocab.len() == deserialized_vocab.len(), "Vocabulary sizes don't match"
    assert vocab.eos_token_id == deserialized_vocab.eos_token_id, "EOS token IDs don't match"

    # Summary
    print("\nSummary:")
    print(f"{'Operation':<20} {'Time (s)':<10} {'Size':<10}")
    print("-" * 40)
    print(f"{'Tokenizer Load':<20} {tokenizer_load_time:<10.2f}")
    print(f"{'Vocab Creation':<20} {vocab_create_time:<10.2f}")
    print(f"{'Serialization':<20} {serialize_time:<10.2f} {file_size:.2f}MB")
    print(f"{'Deserialization':<20} {deserialize_time:<10.2f}")

if __name__ == "__main__":
    test_token_vocabulary_serde("gpt2")
    test_token_vocabulary_serde("teknium/OpenHermes-2.5-Mistral-7B")
    test_token_vocabulary_serde("unsloth/Llama-3.2-1B-Instruct")
