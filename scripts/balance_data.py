import pandas as pd
from sklearn.utils import resample

def balance_data():
    # Load the optimized dataset
    df = pd.read_csv("data/heart_failure_optimized.csv")
    
    # Separate majority and minority classes
    df_majority = df[df.DEATH_EVENT == 0]
    df_minority = df[df.DEATH_EVENT == 1]
    
    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority, 
        replace=True,     # sample with replacement
        n_samples=len(df_majority),    # to match majority class
        random_state=42   # reproducible results
    )
    
    # Combine majority class with upsampled minority class
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save the balanced dataset
    df_balanced.to_csv("data/heart_failure_balanced.csv", index=False)
    print("✅ Balanced dataset saved as 'data/heart_failure_balanced.csv'!")

if __name__ == "__main__":
    balance_data()