def mcculloch_pitts_and_not(a,b):
  w1 = 1
  w2 = -1

  theta = 1

  weighted_sum = a * w1 + b * w2
  if weighted_sum > theta:
    return 1
  else:
    return 0

print("A B | A AND NOT B")
for a in [0,1]:
  for b in [0,1]:
    print(f"{a} {b} | {mcculloch_pitts_and_not(a,b)}")

________________________________________________________________________________________________________________________

