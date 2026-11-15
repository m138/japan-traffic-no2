import requests, json, time, sys, os, zipfile, io
B='https://tenbou.nies.go.jp/download/api/'; OUT=sys.argv[1]; years=sys.argv[2].split(',')
H={'User-Agent':'Mozilla/5.0'}; P=json.dumps([f'{i:02d}' for i in range(1,48)])
def post(u,d,binary=False):
    for k in range(8):
        try:
            r=requests.post(B+u,data=d,headers=H,timeout=90); r.raise_for_status()
            return r.content if binary else r.json()
        except Exception as e: time.sleep(3+3*k)
    raise RuntimeError(u+str(d))
for typ,mat in (('td','["03"]'),('tm','[]')):
    files=post('searchfile.php',{'type':typ,'prefs':P,'years':json.dumps(years),'materials':mat})['files']
    print(typ,len(files),flush=True)
    for f in files:
        fn=f['name']; dst=f'{OUT}/raw/{fn}'
        if not os.path.exists(dst): open(dst,'wb').write(post('archivedownload.php',{'type':typ,'file':fn},True))
        zipfile.ZipFile(dst).extractall(f'{OUT}/x')
print('ok')
