import numpy as np, cv2, subprocess
W,H,N,FPS=640,360,240,24
rng=np.random.default_rng(7)
def noise(sx,sy,seed):
    r=np.random.default_rng(seed).random((sy,sx)).astype(np.float32)
    r=np.tile(r,(1,3))
    n=cv2.resize(r,(W*3,H),interpolation=cv2.INTER_CUBIC)
    n=cv2.GaussianBlur(n,(0,0),25)
    return n[:,W:2*W]
def norm(a): a=a-a.min(); return a/a.max()
layers=[norm(noise(6,3,1)),norm(noise(10,5,2)),norm(noise(14,7,3))]
# sky gradient
y=np.linspace(0,1,H,dtype=np.float32)[:,None,None]
top=np.array([120,130,150],np.float32);bot=np.array([70,125,200],np.float32)  # BGR
sky=top*(1-y)+bot*y*np.ones((1,W,1),np.float32)
# sun glow
yy,xx=np.mgrid[0:H,0:W].astype(np.float32)
glow=np.exp(-(((xx-470)**2+(yy-170)**2)/(2*70**2)))[...,None]*np.array([90,150,235],np.float32)
sky=np.clip(sky+glow,0,255)
# skyline layers
def skyline(seed,minh,maxh,color,wmin,wmax,windows=False):
    r=np.random.default_rng(seed);img=np.zeros((H,W*2,3),np.uint8);m=np.zeros((H,W*2),np.float32);x=0;wins=[]
    while x<W*2:
        w=int(r.integers(wmin,wmax));h=int(r.integers(minh,maxh))
        cv2.rectangle(m,(x,H-h),(x+w,H),1,-1)
        if windows:
            for wx in range(x+4,x+w-5,8):
                for wy in range(H-h+6,H-4,10):
                    if r.random()<.18: wins.append((wx%W,wy,r.random()))
        x+=w+int(r.integers(0,4))
    m=m[:,:W]
    return m,wins
far,_=skyline(11,100,230,0,25,55)
mid,_=skyline(12,80,190,0,30,60)
near,wins=skyline(13,70,170,0,40,80,True)
fogs=[0.55,0.4,0.25]
lights=[(rng.random()*W,H-10-rng.random()*14,rng.choice([1,2]),rng.random()>.5) for _ in range(40)]
parts=[(rng.random()*W,rng.random()*H,rng.choice([1,2]),rng.choice([1,2,3])) for _ in range(160)]
p=subprocess.Popen(['ffmpeg','-y','-f','rawvideo','-pix_fmt','bgr24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-preset','slow','-crf','27','-pix_fmt','yuv420p','-movflags','+faststart','smog.mp4'],stdin=subprocess.PIPE,stderr=subprocess.DEVNULL)
def fog(i,k,spd):
    return np.roll(layers[k],int(W*spd*i/N),axis=1)
for i in range(N):
    f=sky.copy()
    f=f*(1-0.35*fog(i,0,1)[...,None])+np.array([150,165,185],np.float32)*0.35*fog(i,0,1)[...,None]
    for m,col,fg,k,spd in [(far,(95,100,110),0.55,1,-1),(mid,(60,62,72),0.4,2,1),(near,(30,30,38),0.0,2,-1)]:
        mm=m[...,None]
        f=f*(1-mm)+np.array(col,np.float32)*mm
        if fg>0:
            fa=(fog(i,k,spd)[...,None]*0.75+0.15)*fg
            f=f*(1-fa)+np.array([140,160,180],np.float32)*fa
    f=np.clip(f,0,255).astype(np.uint8)
    for (wx,wy,ph) in wins:
        if ((i//(24+int(ph*40))+int(ph*10))%3)!=0 or ph>0.6:
            cv2.rectangle(f,(int(wx),wy),(int(wx)+3,wy+4),(110,200,255),-1)
    # traffic lights (loop)
    for (x,yv,s,red) in lights:
        xx2=int((x+ (W*s*i/N)*(1 if red else -1))%W)
        cv2.circle(f,(xx2,int(yv)),2,(60,60,255) if red else (200,230,255),-1)
    # ground blur
    f[H-14:]=(f[H-14:]*0.5).astype(np.uint8)
    # drifting particles
    ov=np.zeros_like(f)
    for (x,yv,r,s) in parts:
        px=int((x+W*s*i/N)%W);py=int(yv+6*np.sin(2*np.pi*(i/N)*s+x))
        cv2.circle(ov,(px,py),r,(215,225,235),-1)
    f=cv2.addWeighted(f,1,cv2.GaussianBlur(ov,(0,0),1.2),0.45,0)
    # big haze overlay
    hz=(fog(i,2,2)*0.35)[...,None]
    f=(f.astype(np.float32)*(1-hz)+np.array([150,170,190],np.float32)*hz)
    # vignette + grain
    f=f*0.98+rng.normal(0,2,f.shape).astype(np.float32)
    p.stdin.write(np.clip(f,0,255).astype(np.uint8).tobytes())
p.stdin.close();p.wait()
