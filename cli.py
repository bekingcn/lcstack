import sys
sys.path.append("../")

from lcstack.cli import Client
from lcstack import set_config_root
import sys

if __name__ == "__main__":
    set_config_root("examples/configs")
    
    client = Client()
    inputs = client.query
    response = client.complete(inputs=inputs)
    if not isinstance(response, str):
        print("response keys: ", [k for k in response])
    print("Assistant: ", response)