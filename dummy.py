import pandas as pd

# data1 = [1,"Some"]
# data2= [2,"Some other"]

df = pd.DataFrame({
    'Dept': ['IT', 'HR', 'IT', 'Finance'],
    'Salary': [70000, 40000, 80000, 50000]
})

final = df.groupby('Dept')

# Example 2: Compute average salary per department
avg_salary = final['Salary'].mean()
print(avg_salary)


# for data in (final['Dept']):
#     print(data)

# for data in final:
#     print( data['Dept'])



# dataframe1 = pd.DataFrame(data1,columns=["id","nmae"])


# dataframe2 = pd.DataFrame(data2,columns=["id","nmae"])

# dataframe1.merge(dataframe2)

# print(dataframe1)

