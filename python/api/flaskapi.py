from flask import Flask, request, jsonify

app = Flask(__name__)

# Sample in-memory database
employees = [
    {"id": 1, "name": "Tri Le Duc", "department": "OrientSoftware"},
    {"id": 2, "name": "Huy Nguyen Quang", "department": "OrientSoftware"},
    {"id": 3, "name": "Nam Pham Hoang", "department": "People&Culture"}
]

# GET - Retrieve all employees
@app.route('/employees', methods=['GET'])
def get_employees():
    return jsonify(employees)

# GET - Retrieve a single employee by ID
@app.route('/employees/<int:employee_id>', methods=['GET'])
def get_employee(employee_id):
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if employee:
        return jsonify(employee)
    return jsonify({"error": "Employee not found"}), 404

# POST - Add a new employee
@app.route('/employees', methods=['POST'])
def add_employee():
    data = request.json
    if not data or "name" not in data or "department" not in data:
        return jsonify({"error": "Invalid request"}), 400
    
    new_employee = {
        "id": employees[-1]["id"] + 1 if employees else 1,
        "name": data["name"],
        "department": data["department"]
    }
    employees.append(new_employee)
    return jsonify(new_employee), 201

# PUT - Update an existing employee
@app.route('/employees/<int:employee_id>', methods=['PUT'])
def update_employee(employee_id):
    data = request.json
    employee = next((e for e in employees if e["id"] == employee_id), None)
    if not employee:
        return jsonify({"error": "Employee not found"}), 404
    
    employee["name"] = data.get("name", employee["name"])
    employee["department"] = data.get("department", employee["department"])
    return jsonify(employee)

# DELETE - Remove an employee
@app.route('/employees/<int:employee_id>', methods=['DELETE'])
def delete_employee(employee_id):
    global employees
    employees = [e for e in employees if e["id"] != employee_id]
    return jsonify({"message": "Employee deleted"}), 200

if __name__ == '__main__':
    app.run(debug=True, port=8080)