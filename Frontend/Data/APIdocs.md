
# Python API documentation for akashch1512/SingleViewHeigthEstimation
API Endpoints: 5

1. Install the Python client [docs](https://www.gradio.app/guides/getting-started-with-the-python-client) if you don't already have it installed. 

```bash
pip install gradio_client
```

2. Find the API endpoint below corresponding to your desired function in the app. Copy the code snippet, replacing the placeholder values with your own input data. If this is a private Space, you may need to pass your Hugging Face token as well. [Read more](https://www.gradio.app/guides/getting-started-with-the-python-client#connecting-to-a-hugging-face-space).

### API Name: /lambda


```python
from gradio_client import Client

client = Client("akashch1512/SingleViewHeigthEstimation")
result = client.predict(
	api_name="/lambda",
)
print(result)
```


Accepts 0 parameters:



Returns tuple of 6 elements:

[0]: - Type: list[dict(image: filepath, caption: str | None) | dict(video: filepath, caption: str | None)]
- The output value that appears in the "Height & hillshade" Gallery component.

[1]: - Type: filepath
- The output value that appears in the "Height (16-bit)" Image component.

[2]: - Type: str
- The output value that appears in the "value_18" Html component.

[3]: - Type: list[filepath]
- The output value that appears in the "Downloads" File component.

[4]: - Type: str
- The output value that appears in the "value_10" Markdown component.

[5]: - Type: str
- The output value that appears in the "value_11" Markdown component.



### API Name: /predict


```python
from gradio_client import Client, handle_file

client = Client("akashch1512/SingleViewHeigthEstimation")
result = client.predict(
	image_path=handle_file('https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png'),
	gsd=0.5,
	tta=False,
	api_name="/predict",
)
print(result)
```


Accepts 3 parameters:

image_path:
- Type: filepath
- Required
- The input value that is provided in the Image Image component. A local filepath or publicly available URL.

gsd:
- Type: float
- Default: 0.5
- The input value that is provided in the Resolution (m/pixel) Number component. 

tta:
- Type: bool
- Default: False
- The input value that is provided in the Higher quality (slower) Checkbox component. 

Returns tuple of 6 elements:

[0]: - Type: list[dict(image: filepath, caption: str | None) | dict(video: filepath, caption: str | None)]
- The output value that appears in the "Height & hillshade" Gallery component.

[1]: - Type: filepath
- The output value that appears in the "Height (16-bit)" Image component.

[2]: - Type: str
- The output value that appears in the "value_18" Html component.

[3]: - Type: list[filepath]
- The output value that appears in the "Downloads" File component.

[4]: - Type: str
- The output value that appears in the "value_10" Markdown component.

[5]: - Type: str
- The output value that appears in the "value_11" Markdown component.



### API Name: /_mark_warning


```python
from gradio_client import Client

client = Client("akashch1512/SingleViewHeigthEstimation")
result = client.predict(
	text="# Hello!",
	api_name="/_mark_warning",
)
print(result)
```


Accepts 1 parameter:

text:
- Type: str
- Required
- The input value that is provided in the parameter_11 Markdown component. 

Returns 1 element:

- Type: str
- The output value that appears in the "value_11" Markdown component.



### API Name: /lambda_1


```python
from gradio_client import Client

client = Client("akashch1512/SingleViewHeigthEstimation")
result = client.predict(
	api_name="/lambda_1",
)
print(result)
```


Accepts 0 parameters:



Returns 1 element:





### API Name: /lambda_2


```python
from gradio_client import Client

client = Client("akashch1512/SingleViewHeigthEstimation")
result = client.predict(
	api_name="/lambda_2",
)
print(result)
```


Accepts 0 parameters:



Returns tuple of 9 elements:

[0]: - Type: filepath
- The output value that appears in the "Image" Image component.

[1]: - Type: float
- The output value that appears in the "Resolution (m/pixel)" Number component.

[2]: - Type: bool
- The output value that appears in the "Higher quality (slower)" Checkbox component.

[3]: - Type: list[dict(image: filepath, caption: str | None) | dict(video: filepath, caption: str | None)]
- The output value that appears in the "Height & hillshade" Gallery component.

[4]: - Type: filepath
- The output value that appears in the "Height (16-bit)" Image component.

[5]: - Type: str
- The output value that appears in the "value_18" Html component.

[6]: - Type: list[filepath]
- The output value that appears in the "Downloads" File component.

[7]: - Type: str
- The output value that appears in the "value_10" Markdown component.

[8]: - Type: str
- The output value that appears in the "value_11" Markdown component.

