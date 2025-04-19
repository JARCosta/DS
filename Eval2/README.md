
# Encoding

| Class                 | Encoding        |
|-----------------------|-----------------|
| Binary                | Linear Encoding |
| State                 | Dummification   |
| GeneralHealth         | Linear Encoding |
| LastCheckUpTime       | Linear Encoding |
| SmokerStatus          | Taxonomy        |
| ECigaretteUsage       | Taxonomy        |
| RaceEthnicityCategory | Dummification   |
| AgeCategory           | Linear Encoding |
| TetanusLast10Tdap     | Taxonomy        |

# Missing Values

- ## Approach 1
    - Delete any row or column with more than 10% missing values    
    - Fill Missing Values with the most frequent value of the column

- ## Approach 2
    - Delete any row or column with more than 10% missing values    
    - Fill Missing Values with the mean of the 5 nearest neighbors

- ## Another Processes ascertained but not approved
    - Deleting any row with missing values
    - Deleting classes RaceEthnicityCategory or State
        - Both classes can be considered to have overlaping geographic data
    - Deleting classes Sex or HeightInMeters
        - Due to the correlation of 0.85, one of the pair of classes can be deleted, but both present enough extra data to

# Outliers
- ## Approach 1
    - Deleting any outlier row
        - Got a bit worse then no approach
- ## Approach 2
    - Limiting the values of the outliers to to a better fitting range of values (mean +- standard deviation)
        - Gave almost the same results as no approach

# Scaling
- ## Min-Max Approach
    - Scaling the data to a range of 0 to 1
- ## Z-Score Approach
    - Scaling the data through a formula derived from the mean and standard deviation (__x__ - mean) / std

# Balancing
- ## Undersampling
    - Deleting rows from the majority class until the minority class is the same size
- ## Oversampling
    - Duplicating rows from the minority class until the majority class is the same size
- ## SMOTE
    - Creating new rows from the minority class until the majority class is the same size

