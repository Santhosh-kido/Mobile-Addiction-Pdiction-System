import pandas as pd
import numpy as np

def process_data_file(filename):
    df = pd.read_csv(filename)
    
    # Yes/No to 1/0 conversion
    yes_no_cols = ['Use Phone for Class Notes', 'Buy/Access Books from Phone', "Phone's Battery Lasts a Day", 
                   'Run for Charger When Battery Dies', 'Worry About Losing Cell Phone', 'Take Phone to Bathroom',
                   'Use Phone in Social Gatherings', 'Check Phone Before Sleep/After Waking Up', 
                   'Keep Phone Next to While Sleeping', 'Check Emails/Calls/Texts During Class',
                   'Rely on Phone in Awkward Situations', 'On Phone While Watching TV/Eating',
                   'Panic Attack if Phone Left Elsewhere', 'Do you check your phone when spending time with someone',
                   'Live a Day Without Phone', 'Addicted to Phone']
    
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Frequency mapping
    if 'Check Phone Without Notification' in df.columns:
        df['Check Phone Without Notification'] = df['Check Phone Without Notification'].map(
            {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3})
    
    # Gender mapping
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    
    # Gaming hours extraction
    if 'Phone Use for Playing Games' in df.columns:
        df['Phone Use for Playing Games'] = df['Phone Use for Playing Games'].str.extract('(\d+)').astype(float).fillna(0)
    
    # Target mapping
    if 'Target' in df.columns:
        df['Target'] = df['Target'].str.strip().map({'NOT Adicted': 0, 'Maybe addicted': 1, 'Adicted': 2})
    
    # Feature selection
    feature_cols = ['Age', 'Gender', 'Use Phone for Class Notes', 'Buy/Access Books from Phone',
                   "Phone's Battery Lasts a Day", 'Run for Charger When Battery Dies', 'Worry About Losing Cell Phone',
                   'Take Phone to Bathroom', 'Use Phone in Social Gatherings', 'Check Phone Without Notification',
                   'Check Phone Before Sleep/After Waking Up', 'Keep Phone Next to While Sleeping',
                   'Check Emails/Calls/Texts During Class', 'Rely on Phone in Awkward Situations',
                   'On Phone While Watching TV/Eating', 'Panic Attack if Phone Left Elsewhere',
                   'Do you check your phone when spending time with someone', 'Phone Use for Playing Games',
                   'Live a Day Without Phone', 'Addicted to Phone']
    
    # Feature engineering - create composite features
    if 'Target' in df.columns:
        # Create addiction score based on multiple factors
        df['addiction_score'] = (
            df[feature_cols[2:16]].sum(axis=1) +  # Sum of behavioral indicators
            df['Phone Use for Playing Games'] * 0.1  # Gaming hours weight
        )
    
    X = df[feature_cols].fillna(0).values
    y = df['Target'].fillna(0).values if 'Target' in df.columns else None
    
    return X, y

def convert_api_features(input_data):
    frequency_map = {"Never": 0, "Rarely": 1, "Sometimes": 2, "Often": 3}
    
    features = [
        input_data.get('age', 25),
        1 if input_data.get('gender') == "Male" else 0,
        1 if input_data.get('usePhoneForClassNotes') == "Yes" else 0,
        1 if input_data.get('buyBooksFromPhone') == "Yes" else 0,
        1 if input_data.get('batteryLastsDay') == "Yes" else 0,
        1 if input_data.get('runForCharger') == "Yes" else 0,
        1 if input_data.get('worryAboutLosingPhone') == "Yes" else 0,
        1 if input_data.get('takePhoneToBathroom') == "Yes" else 0,
        1 if input_data.get('usePhoneInSocialGatherings') == "Yes" else 0,
        frequency_map.get(input_data.get('checkPhoneWithoutNotification'), 0),
        1 if input_data.get('checkPhoneBeforeSleepAfterWaking') == "Yes" else 0,
        1 if input_data.get('keepPhoneNextToWhileSleeping') == "Yes" else 0,
        1 if input_data.get('checkEmailsCallsTextsDuringClass') == "Yes" else 0,
        1 if input_data.get('relyOnPhoneInAwkwardSituations') == "Yes" else 0,
        1 if input_data.get('onPhoneWhileWatchingTvEating') == "Yes" else 0,
        1 if input_data.get('panicAttackIfPhoneLeftElsewhere') == "Yes" else 0,
        1 if input_data.get('checkPhoneWithSomeone') == "Yes" else 0,
        input_data.get('phoneUseForPlayingGames', 0),
        0 if input_data.get('liveADayWithoutPhone') == "Yes" else 1,
        1 if input_data.get('addictedToPhone') == "Yes" else 0,
    ]
    return features