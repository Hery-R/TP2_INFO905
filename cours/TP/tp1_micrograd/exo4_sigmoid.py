from micrograd.engine import Value

x = Value(0.0)
y = x.sigmoid()
y.backward()

print(f"sigmoid(0) = {y.data}")  # Devrait être 0.5
print(f"d(sigmoid)/dx en 0 = {x.grad}")  # Devrait être 0.25