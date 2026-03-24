import pandas as pd
from googleapiclient.discovery import build
import os

# Paste your API key here
API_KEY = 'AIzaSyAfph2sosAggObSlhRF76gcWNDuYvxhzLs'

def scrape_youtube_comments(video_id, max_comments=1000):
    youtube = build('youtube', 'v3', developerKey=API_KEY)
    
    comments =[]
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    
    print(f"Scraping comments for video: {video_id}...")
    
    while request and len(comments) < max_comments:
        response = request.execute()
        
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100,
                textFormat="plainText"
            )
        else:
            break
            
    # Save to dataframe
    df = pd.DataFrame(comments, columns=['text'])
    
    # Save to your raw data folder
    output_path = f"data/raw/youtube_{video_id}.csv"
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"✅ Saved {len(df)} comments to {output_path}")

if __name__ == "__main__":
    # Example Video ID (Extract from a YouTube URL: https://www.youtube.com/watch?v=VIDEO_ID)
    # Target videos with lots of Roman Urdu (e.g., Coke Studio, Pakistani dramas, Tech reviewers)
    target_video_id = "qak37ots-zg"
    scrape_youtube_comments(target_video_id, max_comments=2000)