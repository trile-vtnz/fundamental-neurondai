import requests

url = "http://127.0.0.1:8080/employees/4"
data = {
    "name": "ABC",
    "department": "OrientSoftware"
}
# response = requests.post(url, json=data)
response = requests.put(url, json=data)
# response = requests.delete(url)

print(response.json())  # Prints the response
