import pandas as pd
import json
import os
import time
from googleapiclient.errors import HttpError
from datetime import datetime, timedelta
import shutil

def load_used_ids(USED_IDS_FILE):
    if os.path.exists(USED_IDS_FILE):
        with open(USED_IDS_FILE, 'r') as f:
            return set(json.load(f))
    return set()

# Fungsi simpan video ID yang digunakan
def save_used_ids(USED_IDS_FILE, used_ids):
    with open(USED_IDS_FILE, 'w') as f:
        json.dump(list(used_ids), f, indent=4)

def search_all_new_videos(youtube, query, used_ids, max_new=10):
    video_ids = []
    next_page_token = None

    while len(video_ids) < max_new:
        try:
            request = youtube.search().list(
                q=query,
                part='snippet',
                type='video',
                maxResults=50,
                pageToken=next_page_token,
                relevanceLanguage='id',
                regionCode='ID'
            )
            response = request.execute()

            for item in response['items']:
                if 'videoId' in item.get('id', {}):
                    vid = item['id']['videoId']
                    if vid not in used_ids and vid not in video_ids:
                        video_ids.append(vid)
                        if len(video_ids) >= max_new:
                            break

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break
        except Exception as e:
            print(f"Error saat mencari video: {e}")
            break

    return video_ids

# Fungsi untuk ambil komentar dari video (termasuk replies)
def get_comments_with_replies(youtube, video_id, max_comments=2000, max_replies_per_comment=2000):
    all_comments = []
    seen_comment_ids = set()
    next_page_token = None

    try:
        while len(all_comments) < max_comments:
            request = youtube.commentThreads().list(
                part='snippet,replies',
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token,
                textFormat='plainText'
            )
            response = request.execute()

            for item in response.get('items', []):
                comment = item['snippet']['topLevelComment']['snippet']
                comment_id = item['snippet']['topLevelComment']['id']

                if comment_id in seen_comment_ids:
                    continue

                main_comment = {
                    'tanggal': datetime.strptime(comment.get('publishedAt'), '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d'),
                    'sentimen': comment.get('textDisplay')
                }
                all_comments.append(main_comment)
                seen_comment_ids.add(comment_id)

                # Ambil replies inline
                inline_replies = item.get('replies', {}).get('comments', [])
                for reply_item in inline_replies:
                    reply = reply_item['snippet']
                    reply_comment = {
                        'tanggal': datetime.strptime(reply.get('publishedAt'), '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d'),
                        'sentimen': reply.get('textDisplay')
                    }
                    all_comments.append(reply_comment)

                # Ambil replies tambahan jika belum lengkap
                total_replies = item['snippet'].get('totalReplyCount', 0)
                if total_replies > len(inline_replies):
                    additional_replies = []
                    reply_next_page = None

                    while len(additional_replies) < max_replies_per_comment:
                        try:
                            reply_request = youtube.comments().list(
                                part='snippet',
                                parentId=comment_id,
                                maxResults=min(100, max_replies_per_comment - len(additional_replies)),
                                pageToken=reply_next_page,
                                textFormat='plainText'
                            )
                            reply_response = reply_request.execute()

                            for reply_item in reply_response.get('items', []):
                                reply = reply_item['snippet']
                                reply_comment = {
                                    'tanggal': datetime.strptime(reply.get('publishedAt'), '%Y-%m-%dT%H:%M:%SZ').strftime('%Y-%m-%d'),
                                    'sentimen': reply.get('textDisplay')
                                }
                                additional_replies.append(reply_comment)

                            reply_next_page = reply_response.get('nextPageToken')
                            if not reply_next_page:
                                break

                            time.sleep(0.1)

                        except Exception as e:
                            print(f"Error saat mengambil reply tambahan untuk comment {comment_id}: {e}")
                            break

                    all_comments.extend(additional_replies)

            next_page_token = response.get('nextPageToken')
            if not next_page_token:
                break

            # Delay antar halaman
            time.sleep(0.2)

    except HttpError as e:
        print(f"Error saat mengambil komentar dari video {video_id}: {e}")
        raise e

    return all_comments

# Fungsi untuk mendapatkan info video
def get_video_info(youtube, video_id):
    try:
        request = youtube.videos().list(
            part='snippet,statistics',
            id=video_id
        )
        response = request.execute()

        if response['items']:
            video = response['items'][0]
            return {
                'video_id': video_id,
                'title': video['snippet'].get('title'),
                'channel': video['snippet'].get('channelTitle'),
                'published_at': video['snippet'].get('publishedAt'),
                'view_count': video['statistics'].get('viewCount'),
                'like_count': video['statistics'].get('likeCount'),
                'comment_count': video['statistics'].get('commentCount')
            }
    except Exception as e:
        print(f"Error saat mengambil info video {video_id}: {e}")

    return None

# Fungsi utama
def main_youtube(youtube, USED_IDS_FILE, query, TICKER):
    max_results = 10
    max_comments_per_video = 10000
    max_replies_per_comment = 5000

    # Load ID yang sudah pernah dipakai
    used_ids = load_used_ids(USED_IDS_FILE)
    new_comments = []
    new_video_info = []
    new_ids = []

    print(f"Mencari video dengan query: '{query}'")
    print(f"Video yang sudah pernah digunakan: {len(used_ids)}")

    # Cari video
    #video_ids = search_videos(query, max_results=max_results)
    video_ids = search_all_new_videos(youtube, query, used_ids, max_new=10)

    if not video_ids:
      print("❌ Tidak ada video baru ditemukan.")
      return

    print(f"✅ Ditemukan {len(video_ids)} video baru untuk diproses.")

    for i, vid in enumerate(video_ids, 1):
        if vid in used_ids:
            print(f"[{i}/{len(video_ids)}] Video ID {vid} sudah pernah digunakan, dilewati.")
            continue

        print(f"[{i}/{len(video_ids)}] Memproses video ID: {vid}")

        # Ambil info video
        video_info = get_video_info(youtube, vid)
        if video_info:
            new_video_info.append(video_info)
            print(f"  - Title: {video_info['title'][:50]}...")
            print(f"  - Channel: {video_info['channel']}")

        # Ambil komentar dan replies
        try:
            comments = get_comments_with_replies(youtube, vid, max_comments_per_video, max_replies_per_comment)

            used_ids.add(vid)

            if comments:
                new_comments.extend(comments)
                new_ids.append(vid)

                print(f"  - Berhasil mengambil {len(comments)} komentar")
            else:
                print(f"  - Tidak ada komentar ditemukan untuk video ini")

        except HttpError as e:
            if e.resp.status == 403 and "commentsDisabled" in str(e):
                print("Komentar pada video ini dinonaktifkan")
                used_ids.add(vid)
            else:
                print(f"Error fatal saat mengambil komentar video {vid}: {e}")

        # Delay antar video
        time.sleep(1)

    # Simpan data baru
    if new_comments:
        # Simpan komentar ke CSV
        df_comments = pd.DataFrame(new_comments)
        df_comments.to_csv(f'data/{TICKER}/data_collection/youtube_comments_{TICKER}.csv', mode='a', header=not os.path.exists(f'data/{TICKER}/data_collection/youtube_comments_{TICKER}.csv'), index=False, encoding='utf-8-sig')

        print(f"\n=== RINGKASAN ===")
        print(f"Berhasil disimpan:")
        print(f"  - {len(new_comments)} total komentar baru didapatkan")
        print(f"  - dari {len(new_ids)} video baru")
        print(f"  - Info {len(new_video_info)} video")
    else:
        print("\nTidak ada video baru ditemukan atau tidak ada komentar yang bisa diambil.")

    # Simpan semua ID yang telah digunakan
    save_used_ids(USED_IDS_FILE, used_ids)
    print(f"\nTotal video yang telah diproses: {len(used_ids)}")


def X_scrapping(twitter_auth_token, search_keyword, start_date, end_date, output_dir, log_dir, TICKER,
                 batch_days=1, batch_limit=100, delay_seconds=30):
    batch_files = []
    current_date = start_date
    final_output_path = f'data/{TICKER}/data_collection/Full_X_comments_{TICKER}.csv'
    amount_data = 0

    while current_date < end_date:
        batch_start = current_date
        batch_end = min(current_date + timedelta(days=batch_days), end_date)

        filename = f"New_X_comments_{TICKER}.csv"
        output_filename = filename

        query = f"{search_keyword} lang:id since:{batch_start.strftime('%Y-%m-%d')} until:{batch_end.strftime('%Y-%m-%d')}"
        log_file = f"{log_dir}/log.txt"

        command = (
            f'npx -y tweet-harvest@2.6.1 '
            f'-o "{output_filename}" '
            f'-s "{query}" '
            f'--tab "LATEST" '
            f'-l {batch_limit} '
            f'--token {twitter_auth_token} '
            f'> {log_file} 2>&1'
        )

        print(f"\n🚀 Menjalankan batch: {batch_start.strftime('%Y-%m-%d')} → {batch_end.strftime('%Y-%m-%d')}")
        exit_code = os.system(command)

        source_path = f"tweets-data/{filename}"
        destination_path = os.path.join(output_dir, filename)
        batch_files.append(destination_path)

        if exit_code != 0:
            print(f"⚠️ Gagal menjalankan tweet-harvest. Cek log: {log_file}")
            break
        else:
            if os.path.exists(source_path):
                shutil.move(source_path, destination_path)
                try:
                    df = pd.read_csv(destination_path)
                    print(f"✅ Data Berhasil disimpan | Jumlah tweet: {len(df)}")

                    amount_data += len(df)

                    # Simpan ke satu file akhir
                    df['created_at'] = pd.to_datetime(df['created_at'], format='%a %b %d %H:%M:%S %z %Y', errors='coerce')
                    df['created_at'] = df['created_at'].dt.strftime('%Y-%m-%d')

                    df_filtered = df[['created_at', 'full_text']].rename(columns={'created_at': 'tanggal', 'full_text': 'sentimen'})
                    df_filtered.to_csv(
                        final_output_path,
                        mode='a',
                        header=not os.path.exists(final_output_path),
                        index=False,
                        encoding='utf-8-sig'
                    )

                except Exception as e:
                    print(f"⚠️ Error membaca CSV: {e}")
            else:
                print(f"⚠️ File tidak ditemukan: {source_path}")

        time.sleep(delay_seconds)
        current_date = batch_end

    print(f"Total sentimen baru yang didapatkan: {amount_data}")