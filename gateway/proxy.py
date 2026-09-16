import asyncio
import base64
import ipaddress
import json
import pathlib
import socket
import sqlite3
import time
import uuid
from contextlib import closing
from typing import Dict
from urllib.parse import unquote_to_bytes, urlparse

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response

app=FastAPI(title="MiniMax H3 Unified Video API",version="3.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_credentials=False,allow_methods=["*"],allow_headers=["*"],expose_headers=["Content-Disposition"])
BACKENDS={"fl2va":"http://h3-fl2va:8091","ref2va":"http://h3-ref2va:8091"}
job_tasks:Dict[str,asyncio.Task]={}
SIZES={
 "480P":{"21:9":(1120,480),"16:9":(864,480),"4:3":(640,480),"1:1":(480,480),"3:4":(480,640),"9:16":(480,864)},
 "768P":{"21:9":(1792,768),"16:9":(1344,768),"4:3":(1024,768),"1:1":(768,768),"3:4":(768,1024),"9:16":(768,1344)}
}

# ---------------------------------------------------------------- storage --
# Job metadata lives in a small file database (SQLite, parameter binding
# only); generated videos are stored as mp4 files next to it. Both survive
# gateway restarts.
DATA_DIR=pathlib.Path("/app/data")
VIDEO_DIR=DATA_DIR/"videos"
DB_PATH=DATA_DIR/"h3_gateway.db"

def init_db():
    DATA_DIR.mkdir(parents=True,exist_ok=True)
    VIDEO_DIR.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS jobs ("
            "id TEXT PRIMARY KEY,"
            "object TEXT NOT NULL,"
            "status TEXT NOT NULL,"
            "progress INTEGER NOT NULL,"
            "created_at INTEGER,"
            "started_at INTEGER,"
            "completed_at INTEGER,"
            "error TEXT,"
            "size INTEGER,"
            "backend TEXT,"
            "video_path TEXT)"
        )
    try:
        with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
            conn.execute("ALTER TABLE jobs ADD COLUMN meta TEXT")
    except sqlite3.OperationalError:
        pass
    # async jobs live only in gateway memory; anything non-terminal at startup
    # was orphaned by the previous process
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
        conn.execute(
            "UPDATE jobs SET status='failed', error=? WHERE status IN ('queued','in_progress')",
            (json.dumps({"message": "网关重启导致任务中断，请重新提交"}, ensure_ascii=False),),
        )

init_db()

def db_insert_job(job:dict):
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
        conn.execute(
            "INSERT INTO jobs (id,object,status,progress,created_at,started_at,completed_at,error,size,backend,video_path,meta)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (job["id"],job["object"],job["status"],job["progress"],job.get("created_at"),job.get("started_at"),
             job.get("completed_at"),json.dumps(job["error"],ensure_ascii=False) if job.get("error") else None,
             job.get("size"),job.get("backend"),job.get("video_path"),job.get("meta")),
        )

def db_mark_running(job_id:str,started_at:int):
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
        conn.execute("UPDATE jobs SET status='in_progress',started_at=? WHERE id=?",(started_at,job_id))

def db_mark_completed(job_id:str,completed_at:int,size:int,video_path:str):
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
        conn.execute("UPDATE jobs SET status='completed',progress=100,completed_at=?,size=?,video_path=? WHERE id=?",
                     (completed_at,size,video_path,job_id))

def db_mark_failed(job_id:str,error:dict):
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
        conn.execute("UPDATE jobs SET status='failed',error=? WHERE id=?",
                     (json.dumps(error,ensure_ascii=False),job_id))

def db_mark_cancelled(job_id:str):
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
        conn.execute("UPDATE jobs SET status='cancelled' WHERE id=?",(job_id,))

def db_delete_job(job_id:str)->bool:
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn,conn:
        cursor=conn.execute("DELETE FROM jobs WHERE id=?",(job_id,))
        return cursor.rowcount>0

def db_get_job(job_id:str)->dict|None:
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn:
        conn.row_factory=sqlite3.Row
        row=conn.execute("SELECT * FROM jobs WHERE id=?",(job_id,)).fetchone()
    return public_row(row) if row else None

def db_list_jobs()->list[dict]:
    with closing(sqlite3.connect(DB_PATH,timeout=30)) as conn:
        conn.row_factory=sqlite3.Row
        rows=conn.execute("SELECT * FROM jobs ORDER BY created_at DESC,id DESC").fetchall()
    return [public_row(row) for row in rows]

def public_row(row:sqlite3.Row)->dict:
    meta=None
    try:
        if row["meta"]: meta=json.loads(row["meta"])
    except (KeyError,IndexError):
        pass
    return {
        "id":row["id"],
        "object":row["object"],
        "status":row["status"],
        "progress":row["progress"],
        "created_at":row["created_at"],
        "started_at":row["started_at"],
        "completed_at":row["completed_at"],
        "error":json.loads(row["error"]) if row["error"] else None,
        "size":row["size"],
        "task":meta.get("task") if meta else None,
        "prompt":meta.get("prompt") if meta else None,
        "params":meta.get("params") if meta else None,
    }

def describe_payload(payload:dict)->dict:
    content=payload.get("content") or []
    texts=[x.get("text","").strip() for x in content if x.get("type")=="text" and isinstance(x.get("text"),str)]
    prompt=" ".join(t for t in texts if t)
    roles=[x.get("role") for x in content if x.get("type")=="image_url"]
    has_ref="reference_image" in roles or any(x.get("type") in ("video_url","audio_url") for x in content)
    if has_ref: task="ref2va"
    elif "last_frame" in roles:
        task="fl2va_first_last" if any(r in (None,"first_frame") for r in roles) else "fl2va_last"
    elif roles: task="fl2va_first"
    else: task="t2va"
    cfg=payload.get("generation_config") or {}
    return {"task":task,"prompt":prompt[:400],
            "params":{"resolution":payload.get("resolution"),"duration":payload.get("duration"),
                      "ratio":payload.get("ratio"),"steps":cfg.get("num_inference_steps",50),"seed":cfg.get("seed",1101)}}

# ------------------------------------------------------------------ helpers --
def err(status:int,message:str,kind:str="bad_request_error"):
    return JSONResponse({"type":"error","error":{"type":kind,"message":message,"http_code":str(status)},"request_id":uuid.uuid4().hex},status_code=status)

def auth_header(request:Request)->str:
    return request.headers.get("authorization","")

def item_url(item:dict,key:str)->str|None:
    value=item.get(key)
    if isinstance(value,str): return value
    if isinstance(value,dict): return value.get("url")
    return None

def assert_safe_reference_url(url:str)->None:
    """Only data URLs and public http(s) URLs may be dereferenced by the server."""
    parsed=urlparse(url)
    if parsed.scheme=="data": return
    if parsed.scheme not in {"http","https"}:
        raise ValueError("reference URL must be a data URL or an http(s) URL")
    hostname=parsed.hostname
    if not hostname:
        raise ValueError("reference URL is missing a host")
    port=parsed.port or (443 if parsed.scheme=="https" else 80)
    try:
        infos=socket.getaddrinfo(hostname,port,type=socket.SOCK_STREAM)
    except OSError:
        raise ValueError(f"cannot resolve reference URL host: {hostname}")
    for info in infos:
        addr=ipaddress.ip_address(info[4][0])
        if (addr.is_private or addr.is_loopback or addr.is_reserved or addr.is_multicast
                or addr.is_link_local or addr.is_unspecified):
            raise ValueError("reference URL must not point at private, loopback, or reserved addresses")

async def fetch_media(url:str)->bytes:
    if url.startswith("data:"):
        head,sep,payload=url.partition(",")
        if not sep: raise ValueError("invalid data URL")
        if "base64" in head.lower():
            try: return base64.b64decode(payload)
            except Exception: raise ValueError("invalid base64 data URL")
        return unquote_to_bytes(payload)
    assert_safe_reference_url(url)
    async with httpx.AsyncClient(timeout=120.0,follow_redirects=True) as client:
        response=await client.get(url)
        if response.status_code>=300:
            raise ValueError(f"failed to download reference media: HTTP {response.status_code}")
        return response.content

async def build_form(payload:dict):
    model=payload.get("model")
    if model not in {"MiniMax-H3","MiniMax-H3-FL2VA","MiniMax-H3-Ref2VA"}:
        raise ValueError("model must be MiniMax-H3, MiniMax-H3-FL2VA, or MiniMax-H3-Ref2VA")
    content=payload.get("content")
    if not isinstance(content,list) or not content:
        raise ValueError("content must be a non-empty array")
    texts=[x.get("text","").strip() for x in content if x.get("type")=="text" and isinstance(x.get("text"),str)]
    if not texts or not texts[0]:
        raise ValueError("content must include a non-empty text item")
    images=[x for x in content if x.get("type")=="image_url"]
    videos=[x for x in content if x.get("type")=="video_url"]
    audios=[x for x in content if x.get("type")=="audio_url"]
    first=[x for x in images if x.get("role") in {None,"first_frame"}]
    last=[x for x in images if x.get("role")=="last_frame"]
    refs=[x for x in images if x.get("role")=="reference_image"]
    if len(first)>1:
        raise ValueError("at most one first_frame image is supported")
    if len(last)>1:
        raise ValueError("at most one last_frame image is supported")
    is_ref=bool(refs or videos or audios or model=="MiniMax-H3-Ref2VA")
    task="ref2va" if is_ref else ("fl2va" if (first or last) else "t2va")
    resolution=str(payload.get("resolution","768P")).upper()
    if resolution=="720P": resolution="768P"
    if resolution=="2K":
        raise ValueError("2K is not enabled on this local L40 deployment")
    if resolution not in SIZES:
        raise ValueError("resolution must be 480P or 768P")
    duration=payload.get("duration")
    if not isinstance(duration,int) or not 4<=duration<=15:
        raise ValueError("duration must be an integer between 4 and 15")
    ratio=payload.get("ratio","adaptive")
    valid={"adaptive","21:9","16:9","4:3","1:1","3:4","9:16"}
    if ratio not in valid: raise ValueError("invalid ratio")
    if task=="t2va" and ratio=="adaptive":
        raise ValueError("ratio is required for text-to-video and cannot be adaptive")
    cfg=payload.get("generation_config") or {}
    if not isinstance(cfg,dict): raise ValueError("generation_config must be an object")
    form={
      "model":"MiniMax-H3-Ref2VA" if is_ref else "MiniMax-H3-FL2VA",
      "prompt":texts[0],
      "fps":"24",
      "num_inference_steps":str(cfg.get("num_inference_steps",50)),
      "flow_shift":str(cfg.get("flow_shift",12)),
      "seed":str(cfg.get("seed",1101)),
    }
    if ratio!="adaptive":
        w,h=SIZES[resolution][ratio];form["width"]=str(w);form["height"]=str(h)
    extra={"task":task,"duration":duration,"audio_flow_shift":cfg.get("audio_flow_shift",3.0)}
    if task=="t2va": extra["aspect_ratio"]=ratio
    keyframe_files=[]
    use_keyframe_files=task=="fl2va" and bool(last)
    if use_keyframe_files:
        extra["frame_indices"]=[0,-1] if first else [-1]
    form["extra_params"]=json.dumps(extra,separators=(",",":"))
    if use_keyframe_files:
        ordered=([first[0]] if first else [])+[last[0]]
        for index,item in enumerate(ordered):
            url=item_url(item,"image_url")
            if not url: raise ValueError("first_frame/last_frame image_url.url is required")
            data=await fetch_media(url)
            keyframe_files.append(("input_references",("keyframe_%d.png"%index,data,"application/octet-stream")))
    elif first:
        url=item_url(first[0],"image_url")
        if not url: raise ValueError("first_frame image_url.url is required")
        assert_safe_reference_url(url)
        form["image_reference"]=json.dumps({"image_url":url},separators=(",",":"))
    if refs:
        url=item_url(refs[0],"image_url")
        if not url: raise ValueError("reference_image image_url.url is required")
        assert_safe_reference_url(url)
        form["image_reference"]=json.dumps({"image_url":url},separators=(",",":"))
    if videos:
        urls=[]
        for x in videos:
            u=item_url(x,"video_url")
            if not u: raise ValueError("video_url.url is required")
            assert_safe_reference_url(u)
            urls.append({"video_url":u})
        form["video_reference"]=json.dumps(urls if len(urls)>1 else urls[0],separators=(",",":"))
    if audios:
        u=item_url(audios[0],"audio_url")
        if not u: raise ValueError("audio_url.url is required")
        assert_safe_reference_url(u)
        form["audio_reference"]=json.dumps({"audio_url":u},separators=(",",":"))
    return ("ref2va" if is_ref else "fl2va"),form,keyframe_files

async def call_backend(payload:dict,authorization:str)->httpx.Response:
    backend,form,keyframe_files=await build_form(payload)
    fields=[(k,(None,v)) for k,v in form.items()]
    fields.extend(keyframe_files)
    async with httpx.AsyncClient(timeout=httpx.Timeout(14400.0,connect=30.0),follow_redirects=True) as client:
        return await client.post(BACKENDS[backend]+"/v1/videos/sync",headers={"Authorization":authorization},files=fields)

async def validate_auth(authorization:str):
    if not authorization: return False
    async with httpx.AsyncClient(timeout=15.0) as client:
        r=await client.get(BACKENDS["fl2va"]+"/v1/models",headers={"Authorization":authorization})
    return r.status_code==200

async def run_job(job_id:str,payload:dict,authorization:str):
    db_mark_running(job_id,int(time.time()))
    try:
        response=await call_backend(payload,authorization)
        if response.status_code>=300:
            try: detail=response.json()
            except Exception: detail=response.text
            db_mark_failed(job_id,{"http_status":response.status_code,"detail":detail})
        else:
            video_path=VIDEO_DIR/(job_id+".mp4")
            await asyncio.to_thread(video_path.write_bytes,response.content)
            db_mark_completed(job_id,int(time.time()),len(response.content),str(video_path))
    except asyncio.CancelledError:
        db_mark_cancelled(job_id);raise
    except Exception as exc:
        db_mark_failed(job_id,{"message":str(exc)})
    finally: job_tasks.pop(job_id,None)

# -------------------------------------------------------------------- routes --
@app.get("/")
async def docs_page(): return FileResponse("/app/docs/index.html",media_type="text/html")

@app.get("/health")
async def health():
    result={}
    async with httpx.AsyncClient(timeout=10.0) as client:
        for name,url in BACKENDS.items():
            try:
                r=await client.get(url+"/health");result[name]={"ok":r.status_code==200,"status":r.status_code}
            except Exception as exc: result[name]={"ok":False,"error":str(exc)}
    status=200 if all(x.get("ok") for x in result.values()) else 503
    return JSONResponse({"status":"ok" if status==200 else "degraded","services":result},status_code=status)

@app.get("/v1/models")
async def models(request:Request):
    if not await validate_auth(auth_header(request)): return err(401,"Invalid or missing API key","authorized_error")
    return {"object":"list","data":[{"id":"MiniMax-H3","object":"model","tasks":["t2va","fl2va","ref2va"]}]}

@app.post("/v1/videos/sync")
async def create_sync(request:Request):
    if request.headers.get("content-type","").split(";")[0]!="application/json": return err(415,"Content-Type must be application/json")
    if not await validate_auth(auth_header(request)): return err(401,"Invalid or missing API key","authorized_error")
    try: payload=await request.json();response=await call_backend(payload,auth_header(request))
    except (ValueError,json.JSONDecodeError) as exc: return err(400,str(exc))
    except Exception as exc: return err(500,str(exc),"server_error")
    return Response(response.content,response.status_code,{"content-type":response.headers.get("content-type","application/json")})

@app.post("/v1/videos")
async def create_video(request:Request):
    if request.headers.get("content-type","").split(";")[0]!="application/json": return err(415,"Content-Type must be application/json")
    if not await validate_auth(auth_header(request)): return err(401,"Invalid or missing API key","authorized_error")
    try:
        payload=await request.json();backend,_,_=await build_form(payload)
    except (ValueError,json.JSONDecodeError) as exc: return err(400,str(exc))
    except Exception as exc: return err(500,str(exc),"server_error")
    job_id="h3_"+uuid.uuid4().hex;now=int(time.time())
    job={"id":job_id,"object":"video","status":"queued","progress":0,"created_at":now,
         "started_at":None,"completed_at":None,"error":None,"size":None}
    meta=describe_payload(payload)
    db_insert_job({**job,"backend":backend,"video_path":None,
                   "meta":json.dumps(meta,ensure_ascii=False)})
    job_tasks[job_id]=asyncio.create_task(run_job(job_id,payload,auth_header(request)))
    return JSONResponse(job,status_code=202)

@app.get("/v1/videos")
async def list_videos(request:Request):
    if not await validate_auth(auth_header(request)): return err(401,"Invalid or missing API key","authorized_error")
    return {"object":"list","data":db_list_jobs()}

@app.get("/v1/videos/{job_id}")
async def get_video(request:Request,job_id:str):
    if not await validate_auth(auth_header(request)): return err(401,"Invalid or missing API key","authorized_error")
    job=db_get_job(job_id)
    return job if job else err(404,"Video job not found","not_found_error")

@app.get("/v1/videos/{job_id}/content")
async def get_content(request:Request,job_id:str):
    if not await validate_auth(auth_header(request)): return err(401,"Invalid or missing API key","authorized_error")
    job=db_get_job(job_id)
    if not job:return err(404,"Video job not found","not_found_error")
    if job["status"]!="completed":return err(409,"Video is not completed: "+job["status"],"conflict_error")
    video_path=pathlib.Path(job.get("video_path") or VIDEO_DIR/(job_id+".mp4"))
    if not video_path.is_file():return err(404,"Video result file is missing","not_found_error")
    return FileResponse(video_path,media_type="video/mp4",filename=job_id+".mp4")

@app.delete("/v1/videos/{job_id}")
async def delete_video(request:Request,job_id:str):
    if not await validate_auth(auth_header(request)): return err(401,"Invalid or missing API key","authorized_error")
    if not db_get_job(job_id):return err(404,"Video job not found","not_found_error")
    task=job_tasks.pop(job_id,None)
    if task and not task.done():task.cancel()
    db_delete_job(job_id)
    try: (VIDEO_DIR/(job_id+".mp4")).unlink()
    except FileNotFoundError: pass
    return Response(status_code=204)
