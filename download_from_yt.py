import csv

import youtube_dl


class MyLogger(object):
    def debug(self, msg):
        pass

    def warning(self, msg):
        pass

    def error(self, msg):
        print(msg)


def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')


def download_music(youtube_id, directory):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'logger': MyLogger(),
        'progress_hooks': [my_hook],
        'outtmpl': directory + '/%(title)s.%(ext)s'
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        print("downloading new music - "+youtube_id)
        ydl.download(['https://www.youtube.com/watch?v='+youtube_id])


def download_musics(musics_id, directory):
    for music_id in musics_id:
        download_music(music_id, directory)


def download_musics_from_csv(csv_file, directory):
    with open(csv_file) as file_obj:
        reader_obj = csv.reader(file_obj)

        # Skips the heading using next() method
        # for i in range(1, 77):
        next(file_obj)

        for row in reader_obj:
            download_music(row[0], directory)


