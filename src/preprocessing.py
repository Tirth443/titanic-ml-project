import pandas as pd


def prepare_data(path):
    df = pd.read_csv(path)

    # Missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    df = df.drop(columns=['Cabin'])

    # Feature engineering
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']

    # Title extraction
    df['Title'] = df['Name'].str.extract(r", ([A-Za-z]+)\.", expand=False)
    df['Title'] = df['Title'].fillna('Unknown')

    df['Title'] = df['Title'].replace(
        ['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],
        'Rare'
    )
    df['Title'] = df['Title'].replace(['Mlle','Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Age group
    df['AgeGroup'] = pd.cut(df['Age'], bins=[0,12,20,40,60,80], labels=[0,1,2,3,4])
    df['AgeGroup'] = df['AgeGroup'].astype(float).fillna(2)

    # Drop unused
    df = df.drop(columns=['Name', 'Ticket', 'PassengerId'])

    # Safe encoding
    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

    title_map = {
        'Mr': 0,
        'Miss': 1,
        'Mrs': 2,
        'Master': 3,
        'Rare': 4,
        'Unknown': 5
    }
    df['Title'] = df['Title'].map(title_map).fillna(5)

    X = df.drop('Survived', axis=1)
    y = df['Survived']

    return X, y