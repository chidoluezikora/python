# Write your code here
# HINT: create a dictionary from flowers.txt
flowers = {}
with open("flowers.txt") as f:
    for line in f:
        if len(line.split()) < 3:
            flowers[line.split()[0][0]] = line.split()[1]
            continue
        flowers[line.split()[0][0]] = line.split()[1] + " " + line.split()[2]
# HINT: create a function to ask for user's first and last name
def generate():
    name = input("Enter your First [space] Last name only: ")
    first_letter = name[0].capitalize()
    return "Unique flower name with the first letter: " + flowers[first_letter]

# print the desired output
print(generate())