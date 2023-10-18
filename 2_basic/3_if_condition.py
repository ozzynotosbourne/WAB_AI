x = input('How old are you? ')
age = int(x)

if age < 18:
    print('You are to young')
else:
    print('Good age')
print('Program ends')

x = input('Howe much do you earn? ')
salary = int(x)

if salary > 5000:
    print(f'You will pay {salary * 0.2} of tax')
else:
    print(f'You will pay {salary * 0.1} of tax')