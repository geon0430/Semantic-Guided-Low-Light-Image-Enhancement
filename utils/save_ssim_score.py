import os
import pandas as pd


def save_ssim_score(
    csv_file_path: str, 
    output_path: str, 
    test_file_name: str, 
    ssim_score: str
) -> None:
    print("start save ssim score to csv")
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.DataFrame(columns=['Process File Name', 'Process Output File Name', 'SSIM Score'])
        
    new_row = pd.DataFrame([{
        'Process File Name': output_path, 
        'Process Output File Name': test_file_name, 
        'SSIM Score': ssim_score
    }])
    
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(csv_file_path, index=False)

