#!/usr/bin/env python3
"""
Comprehensive test of all TTS components in speech-mcp

This script tests all TTS components:
1. Direct TTS utility (direct_tts.py)
2. Direct TTS adapter (direct_adapter.py)
3. OpenAI TTS adapter (openai_adapter.py)

It loads environment variables from .env file to ensure proper configuration.
"""
import os
import sys
import time
import traceback
from dotenv import load_dotenv

# Load environment variables from .env file
print("Loading environment variables from .env file...")
load_dotenv()

# Check for required environment variables
required_vars = [
    'OPENAI_API_KEY',
    'OPENAI_TTS_API_BASE_URL',
    'SPEECH_MCP_TTS_MODEL',
    'SPEECH_MCP_TTS_VOICE'
]

missing_vars = [var for var in required_vars if not os.environ.get(var)]
if missing_vars:
    print(f"ERROR: Missing required environment variables: {', '.join(missing_vars)}")
    print("Please make sure these are set in your .env file or in your environment.")
    sys.exit(1)

# Print banner and configuration
print("="*50)
print(" SPEECH-MCP TEXT-TO-SPEECH TEST SUITE ")
print("="*50)
print("\nEnvironment Configuration:")
print(f"OPENAI_API_KEY: {'[SET]' if os.environ.get('OPENAI_API_KEY') else '[NOT SET]'}")
print(f"OPENAI_TTS_API_BASE_URL: {os.environ.get('OPENAI_TTS_API_BASE_URL')}")
print(f"SPEECH_MCP_TTS_MODEL: {os.environ.get('SPEECH_MCP_TTS_MODEL')}")
print(f"SPEECH_MCP_TTS_VOICE: {os.environ.get('SPEECH_MCP_TTS_VOICE')}")

# Test results tracking
test_results = []

def run_test(test_name, test_func, *args, **kwargs):
    """Run a test and track its result"""
    print(f"\n[TEST] {test_name}")
    print("-" * (7 + len(test_name)))
    start_time = time.time()
    success = False
    error = None
    
    try:
        success = test_func(*args, **kwargs)
    except Exception as e:
        error = str(e)
        traceback.print_exc()
    
    duration = time.time() - start_time
    result = {
        "name": test_name,
        "success": success,
        "error": error,
        "duration": duration
    }
    test_results.append(result)
    
    print(f"Result: {'✅ PASSED' if success else '❌ FAILED'}")
    if error:
        print(f"Error: {error}")
    print(f"Duration: {duration:.2f} seconds")
    
    return success

# Test 1: Direct TTS utility
def test_direct_tts_utility():
    """Test the direct_tts.py utility"""
    try:
        # Import the direct TTS utility
        from speech_mcp.utils.direct_tts import TextToSpeech, debug_print
        
        print("Creating TextToSpeech instance...")
        direct_tts = TextToSpeech()
        
        # Test speak method
        print("Testing speak method...")
        success = direct_tts.speak("This is a test of the direct TTS utility. If you can hear this, the direct TTS utility is working!")
        print(f"TextToSpeech.speak result: {success}")
        
        # Wait for speech to complete
        time.sleep(3)
        
        return True
    except Exception as e:
        print(f"Error testing direct_tts.py: {e}")
        traceback.print_exc()
        return False

# Test 2: Direct TTS adapter
def test_direct_tts_adapter():
    """Test the direct_adapter.py adapter"""
    try:
        # Import the DirectTTS adapter
        from speech_mcp.tts_adapters.direct_adapter import DirectTTS
        
        print("Creating DirectTTS adapter instance...")
        adapter = DirectTTS()
        
        # Test speak method
        print("Testing speak method...")
        success = adapter.speak("This is a test of the Direct TTS adapter. If you can hear this, the Direct TTS adapter is working!")
        print(f"DirectTTS adapter speak result: {success}")
        
        # Wait for speech to complete
        time.sleep(3)
        
        # Test save_to_file
        output_file = os.path.join(os.getcwd(), "direct_adapter_output.wav")
        print(f"\nTesting save_to_file to {output_file}...")
        save_result = adapter.save_to_file("This is a test of saving audio with the Direct TTS adapter.", output_file)
        print(f"Save to file result: {save_result}")
        
        # Check file
        if save_result and os.path.exists(output_file):
            print(f"File created: {output_file}, size: {os.path.getsize(output_file)} bytes")
            
            # Try playing it
            if sys.platform == "darwin":
                print("Playing saved file...")
                os.system(f"afplay \"{output_file}\"")
            
            # Clean up
            os.unlink(output_file)
            print("Deleted test file")
        
        return success
    except Exception as e:
        print(f"Error testing Direct TTS adapter: {e}")
        traceback.print_exc()
        return False

# Test 3: OpenAI TTS adapter
def test_openai_tts_adapter():
    """Test the openai_adapter.py adapter"""
    try:
        # Import the OpenAITTS adapter
        from speech_mcp.tts_adapters.openai_adapter import OpenAITTS
        
        print("Creating OpenAITTS adapter instance...")
        adapter = OpenAITTS()
        
        # Test speak method
        print("Testing speak method...")
        success = adapter.speak("This is a test of the OpenAI TTS adapter. If you can hear this, the OpenAI TTS adapter is working!")
        print(f"OpenAITTS adapter speak result: {success}")
        
        # Wait for speech to complete
        time.sleep(3)
        
        # Test save_to_file
        output_file = os.path.join(os.getcwd(), "openai_adapter_output.wav")
        print(f"\nTesting save_to_file to {output_file}...")
        save_result = adapter.save_to_file("This is a test of saving audio with the OpenAI TTS adapter.", output_file)
        print(f"Save to file result: {save_result}")
        
        # Check file
        if save_result and os.path.exists(output_file):
            print(f"File created: {output_file}, size: {os.path.getsize(output_file)} bytes")
            
            # Try playing it
            if sys.platform == "darwin":
                print("Playing saved file...")
                os.system(f"afplay \"{output_file}\"")
            
            # Clean up
            os.unlink(output_file)
            print("Deleted test file")
        
        return success
    except Exception as e:
        print(f"Error testing OpenAI TTS adapter: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests and print summary"""
    # Run tests
    run_test("Direct TTS Utility", test_direct_tts_utility)
    run_test("Direct TTS Adapter", test_direct_tts_adapter)
    run_test("OpenAI TTS Adapter", test_openai_tts_adapter)
    
    # Print summary
    print("\n" + "="*50)
    print(" TEST RESULTS SUMMARY ")
    print("="*50)
    
    for result in test_results:
        status = "✅ PASSED" if result["success"] else "❌ FAILED"
        print(f"{result['name']}: {status} ({result['duration']:.2f}s)")
        if result["error"]:
            print(f"  Error: {result['error']}")
    
    # Overall result
    all_passed = all(result["success"] for result in test_results)
    passed_count = sum(1 for result in test_results if result["success"])
    total_count = len(test_results)
    
    print("\nOverall Result:")
    print(f"Passed: {passed_count}/{total_count}")
    print(f"Status: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())