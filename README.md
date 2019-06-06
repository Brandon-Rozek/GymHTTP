# HTTP interface for OpenAI Gym
This library adds a HTTP interface for the [OpenAI Gym Project](https://github.com/openai/gym). Hopefully you will be able to use it in your reinforcement learning projects without noticing!

Why would I want to do this? If you want to decouple the processing of the environment from the training of your models this might be beneficial.

To start the webserver
```bash
export FLASK_APP=gymserver.py
flask run
```

In your main application
```python
from gymclient import Environment
env = Environment("127.0.0.1", 5000)
```
