const $=(selector)=>document.querySelector(selector);

const state={
  sourceHint:"rubbing",
  libraries:{rubbing:[],nmns:[]},
  specimens:[],
  specimen:null,
  document:null,
  audioContext:null,
  audioLibrary:[],
  audioPreviewObjectUrl:null,
  audio:null,
  audioResult:null,
  audioView:"disturbed",
  analysisView:"primary",
  analysisPreviews:{},
  runtime:{
    rubbing:{resolutions:[2048,3072,4096,6144,8192],max_resolution:8192,safe_mode:false},
    nmns:{resolutions:[2048,4096,8192,16384]},
  },
};

const leafFile=$("#leaf-file");
const leafSelect=$("#leaf-select");
const textInput=$("#text-input");
const resolutionSelect=$("#resolution-select");
const analyzeLeafButton=$("#analyze-leaf");
const capacityButton=$("#check-capacity");
const exportButton=$("#export-jpg");
const soundFile=$("#sound-file");
const soundLibrary=$("#sound-library");
const soundPreview=$("#sound-preview");
const applySoundButton=$("#apply-sound");

function sleep(milliseconds){return new Promise((resolve)=>setTimeout(resolve,milliseconds));}

async function jsonRequest(path,options={}){
  const response=await fetch(path,options);
  const payload=await response.json().catch(()=>({}));
  if(!response.ok)throw new Error(payload.error||"Request failed");
  return payload;
}

function fileDataUrl(file){
  return new Promise((resolve,reject)=>{
    const reader=new FileReader();
    reader.addEventListener("load",()=>resolve(reader.result));
    reader.addEventListener("error",()=>reject(new Error("Could not read the image")));
    reader.readAsDataURL(file);
  });
}

function percentile(values,ratio){
  if(!values.length)return 0;
  const ordered=[...values].sort((a,b)=>a-b);
  return ordered[Math.max(0,Math.min(ordered.length-1,Math.floor((ordered.length-1)*ratio)))];
}

function normalizeFrames(frames,key){
  const values=frames.map((frame)=>frame[key]);
  const low=percentile(values,.06);
  const high=percentile(values,.96);
  const span=Math.max(1e-8,high-low);
  frames.forEach((frame)=>{frame[key]=Math.max(0,Math.min(1,(frame[key]-low)/span));});
}

function formatDuration(seconds){
  const minutes=Math.floor(seconds/60);
  const remainder=Math.round(seconds-minutes*60);
  return `${minutes}:${String(remainder).padStart(2,"0")}`;
}

function resetAudioResult(){
  state.audioResult=null;
  state.audioView="disturbed";
  $("#sound-result").hidden=true;
  $("#audio-progress").hidden=true;
  $("#audio-progress-bar").value=0;
  document.querySelectorAll("[data-monitor]").forEach((button)=>button.classList.toggle("active",button.dataset.monitor==="disturbed"));
}

function clearSoundPreview(){
  soundPreview.pause();
  soundPreview.removeAttribute("src");
  soundPreview.load();
  if(state.audioPreviewObjectUrl){
    URL.revokeObjectURL(state.audioPreviewObjectUrl);
    state.audioPreviewObjectUrl=null;
  }
  $("#sound-preview-wrap").hidden=true;
}

function setSoundPreview(source,name){
  clearSoundPreview();
  if(typeof source==="string")soundPreview.src=source;
  else{
    state.audioPreviewObjectUrl=URL.createObjectURL(source);
    soundPreview.src=state.audioPreviewObjectUrl;
  }
  $("#sound-preview-name").textContent=name;
  $("#sound-preview-wrap").hidden=false;
  soundPreview.load();
}

function drawSoundWave(){
  const canvas=$("#sound-wave");
  const width=Math.max(1,canvas.clientWidth);
  const height=Math.max(1,canvas.clientHeight);
  const ratio=Math.min(2,window.devicePixelRatio||1);
  canvas.width=Math.floor(width*ratio);canvas.height=Math.floor(height*ratio);
  const context=canvas.getContext("2d");
  context.scale(ratio,ratio);
  context.fillStyle="#0b1511";context.fillRect(0,0,width,height);
  context.strokeStyle="#213a2d";
  for(let row=1;row<4;row+=1){context.beginPath();context.moveTo(0,height*row/4);context.lineTo(width,height*row/4);context.stroke();}
  if(!state.audio)return;
  const palette={energy:"#74c895",peak:"#d2ad52",noise:"#77a9b7"};
  Object.entries(palette).forEach(([key,color])=>{
    context.strokeStyle=color;context.lineWidth=1;context.beginPath();
    state.audio.frames.forEach((frame,index)=>{
      const x=index/Math.max(1,state.audio.frames.length-1)*width;
      const y=height-7-frame[key]*(height-14);
      if(index===0)context.moveTo(x,y);else context.lineTo(x,y);
    });
    context.stroke();
  });
  context.strokeStyle="#c6525f";
  state.audio.frames.forEach((frame,index)=>{
    if(!frame.impact)return;
    const x=index/Math.max(1,state.audio.frames.length-1)*width;
    context.beginPath();context.moveTo(x,0);context.lineTo(x,12+frame.impact*22);context.stroke();
  });
}

async function analyzeWav(file){
  resetAudioResult();
  applySoundButton.disabled=true;
  $("#sound-file-label").textContent=`Reading ${file.name}…`;
  try{
    const AudioContextClass=window.AudioContext||window.webkitAudioContext;
    if(!AudioContextClass)throw new Error("This browser cannot analyze WAV audio.");
    state.audioContext??=new AudioContextClass();
    const buffer=await state.audioContext.decodeAudioData((await file.arrayBuffer()).slice(0));
    const channels=buffer.numberOfChannels;
    const sampleRate=buffer.sampleRate;
    const stride=Math.max(1,Math.floor(sampleRate/24000));
    const effectiveRate=sampleRate/stride;
    const frameSize=1024;
    const hop=512;
    const frameCount=Math.max(1,Math.floor((buffer.length/stride-frameSize)/hop)+1);
    const frames=[];
    let low=0,lastEnergy=0,lastNoise=0;
    const lowAlpha=Math.min(1,(2*Math.PI*220)/effectiveRate);
    for(let frameIndex=0;frameIndex<frameCount;frameIndex+=1){
      const base=frameIndex*hop*stride;
      let squares=0,lowSquares=0,highSquares=0,peak=0,crossings=0,previous=0;
      for(let index=0;index<frameSize;index+=1){
        const sourceIndex=base+index*stride;
        if(sourceIndex>=buffer.length)break;
        let sample=0;
        for(let channel=0;channel<channels;channel+=1)sample+=buffer.getChannelData(channel)[sourceIndex]/channels;
        low+=lowAlpha*(sample-low);
        const high=sample-low;
        squares+=sample*sample;lowSquares+=low*low;highSquares+=high*high;
        peak=Math.max(peak,Math.abs(sample));
        if((sample>=0)!==(previous>=0))crossings+=1;
        previous=sample;
      }
      const energy=Math.sqrt(squares/frameSize);
      const noise=Math.sqrt(highSquares/frameSize);
      const flux=Math.max(0,energy-lastEnergy)+Math.max(0,noise-lastNoise)*.55;
      frames.push({time:frameIndex*hop/effectiveRate,energy,peak,bass:Math.sqrt(lowSquares/frameSize),brightness:Math.min(1,crossings/frameSize*17),noise,flux,impact:0});
      lastEnergy=energy;lastNoise=noise;
    }
    ["energy","peak","bass","brightness","noise","flux"].forEach((key)=>normalizeFrames(frames,key));
    const fluxGate=percentile(frames.map((frame)=>frame.flux),.9);
    const energyGate=percentile(frames.map((frame)=>frame.energy),.4);
    const gap=Math.max(2,Math.floor(.075*effectiveRate/hop));
    let lastImpact=-gap,impacts=0;
    for(let index=1;index<frames.length-1;index+=1){
      const frame=frames[index];
      if(frame.flux>=fluxGate&&frame.energy>=energyGate&&frame.flux>=frames[index-1].flux&&frame.flux>=frames[index+1].flux&&index-lastImpact>=gap){
        frame.impact=Math.max(frame.flux,frame.peak*.82);lastImpact=index;impacts+=1;
      }
    }
    state.audio={name:file.name,frames,impacts,duration:buffer.duration,sampleRate};
    $("#sound-file-label").textContent=file.name;
    $("#sound-summary").textContent=`${formatDuration(buffer.duration)} · ${frames.length.toLocaleString()} frames · ${impacts} impacts`;
    applySoundButton.disabled=!state.document?.exportable||state.sourceHint!=="rubbing";
    drawSoundWave();
  }catch(error){
    state.audio=null;
    $("#sound-file-label").textContent="Could not read WAV";
    $("#sound-summary").textContent=error.message;
  }
}

async function loadAudioLibrary(){
  const archive=await jsonRequest("/static/archive/archive.json");
  const unique=new Map();
  (archive.entries||[]).forEach((entry)=>{
    if(entry.audio&&entry.sound&&!unique.has(entry.sound)){
      unique.set(entry.sound,{
        name:entry.sound,
        url:entry.audio,
        duration:Number(entry.metrics?.duration||0),
      });
    }
  });
  state.audioLibrary=[...unique.values()];
  soundLibrary.replaceChildren(new Option("Choose from the rubbing archive", ""));
  state.audioLibrary.forEach((recording,index)=>{
    const duration=recording.duration?` · ${formatDuration(recording.duration)}`:"";
    soundLibrary.append(new Option(`${recording.name}${duration}`,String(index)));
  });
  soundLibrary.disabled=!state.audioLibrary.length;
}

async function analyzeLibraryAudio(index){
  const recording=state.audioLibrary[Number(index)];
  if(!recording)return;
  soundFile.value="";
  setSoundPreview(recording.url,recording.name);
  soundLibrary.disabled=true;
  applySoundButton.disabled=true;
  $("#sound-file-label").textContent="Upload a WAV";
  $("#sound-summary").textContent=`Loading ${recording.name}…`;
  try{
    const response=await fetch(recording.url);
    if(!response.ok)throw new Error("Could not load this archive recording.");
    const blob=await response.blob();
    await analyzeWav(new File([blob],recording.name,{type:blob.type||"audio/mpeg"}));
  }catch(error){
    state.audio=null;
    $("#sound-summary").textContent=error.message;
  }finally{
    soundLibrary.disabled=false;
  }
}

function quantizedAudio(levels=16,maxFrames=2400){
  if(!state.audio)return [];
  const frames=state.audio.frames;
  const group=Math.max(1,Math.ceil(frames.length/maxFrames));
  const compact=[];
  for(let start=0;start<frames.length;start+=group){
    const slice=frames.slice(start,start+group);
    const average=(key)=>slice.reduce((sum,frame)=>sum+frame[key],0)/slice.length;
    const maximum=(key)=>Math.max(...slice.map((frame)=>frame[key]));
    compact.push({
      time:slice[0].time,
      energy:average("energy"),peak:maximum("peak"),bass:average("bass"),
      brightness:average("brightness"),noise:average("noise"),impact:maximum("impact"),
    });
  }
  return compact.map((frame)=>({
    time:frame.time,
    energy:Math.round(frame.energy*(levels-1)),peak:Math.round(frame.peak*(levels-1)),
    bass:Math.round(frame.bass*(levels-1)),brightness:Math.round(frame.brightness*(levels-1)),
    noise:Math.round(frame.noise*(levels-1)),impact:frame.impact>0?Math.max(1,Math.round(frame.impact*(levels-1))):0,
  }));
}

function progressElements(kind){
  return {box:$(`#${kind}-progress`),label:$(`#${kind}-progress-label`),value:$(`#${kind}-progress-value`),bar:$(`#${kind}-progress-bar`)};
}

function setProgress(kind,progress,stage){
  const elements=progressElements(kind);
  elements.box.hidden=false;
  elements.bar.value=Math.max(Number(elements.bar.value)||0,Number(progress)||0);
  elements.value.textContent=`${Math.round(elements.bar.value)}%`;
  elements.label.textContent=stage;
}

async function pollJob(identifier,kind){
  const started=Date.now();
  let visual=3;
  while(Date.now()-started<240000){
    const job=await jsonRequest(`/api/jobs/${encodeURIComponent(identifier)}`);
    visual=Math.max(visual,Number(job.progress)||0);
    if(job.status==="running"&&visual<92)visual=Math.min(92,visual+1.4);
    setProgress(kind,visual,job.stage||"Working");
    if(job.status==="complete"){
      setProgress(kind,100,"Ready");
      return job.result;
    }
    if(job.status==="error")throw new Error(job.error||"Analysis failed");
    await sleep(450);
  }
  throw new Error("Analysis took too long. Please try again.");
}

function setStep(step){
  document.querySelectorAll(".journey span").forEach((item)=>item.classList.toggle("active",Number(item.dataset.step)<=step));
}

function resetDecision(){
  state.document=null;
  resetAudioResult();
  $("#capacity-result").hidden=true;
  $("#capacity-progress").hidden=true;
  $("#capacity-progress-bar").value=0;
  $("#export-panel").classList.add("locked");
  $("#export-panel").hidden=true;
  $("#encoded-preview").removeAttribute("src");
  exportButton.disabled=true;
  $("#export-status").textContent="Run the capacity check first.";
  $("#sound-panel").hidden=true;
  $("#sound-panel").classList.add("locked");
  applySoundButton.disabled=true;
  setStep(state.specimen?3:1);
}

function showAnalysisView(view){
  const previews=state.analysisPreviews||{};
  const fallback=previews.primary||previews.overlay||previews.original||state.specimen?.source_url;
  const selected=previews[view]||fallback;
  if(!selected)return;
  state.analysisView=view;
  document.querySelectorAll("[data-analysis-view]").forEach((button)=>{
    button.classList.toggle("active",button.dataset.analysisView===view);
    button.hidden=!previews[button.dataset.analysisView]&&button.dataset.analysisView!=="original";
  });
  $("#analysis-preview").src=selected.startsWith("data:")?selected:`${selected}?v=${Date.now()}`;
}

function renderAnalysis(specimen,analysis={}){
  const panel=$("#analysis-panel");
  const nmns=state.sourceHint==="nmns";
  panel.hidden=!nmns;
  if(!nmns)return;
  state.analysisPreviews={...(specimen.analysis_urls||{}),...(analysis.previews||{})};
  if(!state.analysisPreviews.original)state.analysisPreviews.original=specimen.source_url;
  $("#analysis-isolation").textContent=analysis.isolation_confidence!=null?`${Number(analysis.isolation_confidence).toFixed(0)}%`:"Saved";
  $("#analysis-venation").textContent=(analysis.venation_type||specimen.venation_type||"unknown").replace(/^./,(value)=>value.toUpperCase());
  const confidence=analysis.main_path_confidence??specimen.main_path_confidence;
  $("#analysis-confidence").textContent=confidence==null?"—":`${Number(confidence).toFixed(0)}%`;
  $("#analysis-edges").textContent=analysis.graph_edges??specimen.graph_edges??"—";
  $("#analysis-status").textContent=specimen.ready===false?"Review the primary vein":"Ready for text";
  showAnalysisView(state.analysisPreviews.primary?"primary":state.analysisPreviews.overlay?"overlay":"original");
}

function analysisSettings(){
  return {
    max_side:Number($("#analysis-resolution").value),
    sensitivity:Number($("#analysis-sensitivity").value),
    contrast_scale:Number($("#analysis-contrast").value),
    min_component:Number($("#analysis-component").value),
  };
}

async function reanalyzeLeaf(){
  if(state.sourceHint!=="nmns"||!state.specimen)return;
  const button=$("#reanalyze-leaf");
  button.disabled=true;$("#analysis-loading").hidden=false;$("#analysis-status").textContent="Analyzing…";
  try{
    const payload=await jsonRequest("/api/export",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({filename:state.specimen.id,settings:analysisSettings()}),
    });
    const result=payload.result;
    state.analysisPreviews=result.previews;
    state.specimen.graph_edges=result.graph.stats.edge_count;
    state.specimen.branch_anchors=result.graph.stats.secondary_anchor_count;
    state.specimen.venation_type=result.graph.venation?.type;
    state.specimen.main_path_confidence=result.graph.venation?.main_path_confidence;
    renderAnalysis(state.specimen,{
      previews:result.previews,graph_edges:result.graph.stats.edge_count,
      venation_type:result.graph.venation?.type,main_path_confidence:result.graph.venation?.main_path_confidence,
    });
    $("#analysis-status").textContent="Analysis updated";
  }catch(error){$("#analysis-status").textContent=error.message;}
  finally{button.disabled=false;$("#analysis-loading").hidden=true;}
}

function setResolutionOptions(){
  const runtime=state.runtime[state.sourceHint]||{};
  const values=runtime.resolutions||(state.sourceHint==="nmns"?[2048,4096,8192,16384]:[2048,3072,4096,6144,8192]);
  const maxResolution=state.sourceHint==="nmns"?Math.max(...values):(runtime.max_resolution||Math.max(...values));
  const preferred=state.sourceHint==="nmns"?8192:Math.min(3072,maxResolution);
  resolutionSelect.innerHTML=values.map((value)=>{
    const unavailable=value>maxResolution;
    return `<option value="${value}" ${value===preferred?"selected":""} ${unavailable?"disabled":""}>${value} px${unavailable?" · unavailable online":""}</option>`;
  }).join("");
}

function libraryOption(specimen){
  const option=document.createElement("option");
  option.value=specimen.id;
  option.textContent=specimen.name;
  return option;
}

function refreshLibrary(selected=""){
  leafSelect.replaceChildren();
  state.specimens.forEach((specimen)=>leafSelect.append(libraryOption(specimen)));
  if(selected&&state.specimens.some((item)=>item.id===selected))leafSelect.value=selected;
  $("#library-count").textContent=`${state.specimens.length} saved ${state.specimens.length===1?"leaf":"leaves"}`;
}

function renderLeaf(specimen,analysis=null){
  state.specimen=specimen;
  leafSelect.value=specimen.id;
  $("#leaf-result").hidden=false;
  $("#leaf-preview").src=`${specimen.source_url}?v=${Date.now()}`;
  $("#leaf-preview-link").href=specimen.source_url;
  $("#leaf-name").textContent=specimen.name;
  const nmns=state.sourceHint==="nmns";
  $("#leaf-kind").textContent=nmns?"Natural vein graph carrier":"Pixel and colour carrier";
  const ready=specimen.ready!==false;
  $("#leaf-ready").textContent=ready?"READY":"REVIEW";
  $("#leaf-ready").classList.toggle("review",!ready);
  if(nmns){
    $("#leaf-quality").textContent=analysis?.graph_edges??specimen.graph_edges??"—";
    $("#leaf-quality-label").textContent="Vein edges";
    $("#leaf-texture").textContent=analysis?.branch_anchors??specimen.branch_anchors??"—";
    $("#leaf-texture-label").textContent="Branches";
  }else{
    const quality=analysis?.quality_score??specimen.quality_score;
    const texture=analysis?.texture_coverage??specimen.texture_coverage;
    $("#leaf-quality").textContent=quality==null?"Curated":Number(quality).toFixed(0);
    $("#leaf-quality-label").textContent="Quality";
    $("#leaf-texture").textContent=texture==null?"—":`${(Number(texture)*100).toFixed(0)}%`;
    $("#leaf-texture-label").textContent="Texture";
  }
  $("#leaf-size").textContent=`${specimen.width}×${specimen.height}`;
  renderAnalysis(specimen,analysis||{});
  $("#text-panel").classList.toggle("locked",!ready);
  textInput.disabled=!ready;
  resolutionSelect.disabled=!ready;
  capacityButton.disabled=!ready;
  resetDecision();
}

function choosePipeline(sourceHint){
  state.sourceHint=sourceHint;
  document.body.classList.toggle("nmns-mode",sourceHint==="nmns");
  state.specimens=state.libraries[sourceHint]||[];
  state.specimen=null;
  document.querySelectorAll("[data-source]").forEach((item)=>item.classList.toggle("selected",item.dataset.source===sourceHint));
  setResolutionOptions();
  const preferred=sourceHint==="rubbing"
    ? (state.specimens.find((item)=>item.id==="1558")?.id||state.specimens[0]?.id)
    : (state.specimens.find((item)=>item.id.includes("Acanthopanax"))?.id||state.specimens[0]?.id);
  refreshLibrary(preferred);
  if(leafSelect.value){
    const specimen=state.specimens.find((item)=>item.id===leafSelect.value);
    if(specimen)renderLeaf(specimen);
  }else{
    $("#leaf-result").hidden=true;
    $("#analysis-panel").hidden=true;
    $("#text-panel").classList.add("locked");
    textInput.disabled=true;resolutionSelect.disabled=true;capacityButton.disabled=true;
    resetDecision();
  }
  $("#file-label").textContent="Choose an image";
  $("#mode-eyebrow").textContent=sourceHint==="nmns"?"NMNS VEIN GRAPH MODE":"LEAF TYPOGRAPHY";
  $("#mode-description").textContent=sourceHint==="nmns"
    ? "Analyze a scan. Fill its natural veins. Export a reversible PNG."
    : "Upload a leaf. Check capacity. Export.";
  $("#export-kind").textContent=sourceHint==="nmns"?"Reversible vein PNG":"Reversible pixel JPG";
  exportButton.textContent=sourceHint==="nmns"?"Export PNG":"Export JPG";
  $("#sound-panel").hidden=true;
  $("#decoder-status").textContent="";
}

async function loadLibraries(){
  const [rubbing,runtime]=await Promise.all([
    jsonRequest("/api/colour/specimens"),
    jsonRequest("/api/runtime"),
  ]);
  state.runtime={...state.runtime,...runtime};
  state.libraries.rubbing=(rubbing.specimens||[]).map((item)=>({...item,pipeline:"rubbing"}));
  state.libraries.nmns=[];
  choosePipeline("rubbing");
}

async function uploadLeaf(){
  const file=leafFile.files[0];
  if(!file){$("#file-label").textContent="Choose an image first";return;}
  if(file.size>32*1024*1024){$("#file-label").textContent="Image is over 32 MB";return;}
  analyzeLeafButton.disabled=true;
  setProgress("leaf",2,"Uploading");
  try{
    const data=await fileDataUrl(file);
    const queued=await jsonRequest("/api/jobs/specimen",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({filename:file.name,data,source_hint:state.sourceHint}),
    });
    const result=await pollJob(queued.job_id,"leaf");
    const specimen={...result.specimen,pipeline:state.sourceHint};
    const library=state.libraries[state.sourceHint];
    const existing=library.findIndex((item)=>item.id===specimen.id);
    if(existing>=0)library[existing]=specimen;else library.push(specimen);
    state.specimens=library;
    refreshLibrary(specimen.id);
    renderLeaf(specimen,result.analysis);
    $("#file-label").textContent=result.message;
    leafFile.value="";
  }catch(error){
    setProgress("leaf",100,"Could not analyze");
    $("#file-label").textContent=error.message;
  }finally{
    analyzeLeafButton.disabled=false;
  }
}

function renderCapacity(data){
  state.document=data;
  const selected=data.selected;
  const capacity=data.capacity;
  const ready=Boolean(data.exportable);
  const nmns=state.sourceHint==="nmns";
  $("#capacity-result").hidden=false;
  const resourceLimited=Boolean(selected.resource_limited);
  $("#capacity-badge").textContent=ready?"READY":resourceLimited?"SERVER LIMIT":"TOO LARGE";
  $("#capacity-badge").classList.toggle("fail",!ready);
  $("#capacity-title").textContent=ready?"Ready to export":resourceLimited?"Choose a smaller output":"More space needed";
  $("#capacity-note").textContent=ready
    ? `${selected.resolution}px safely carries this text.`
    : resourceLimited
      ? capacity.runtime_limit?.message||capacity.recommendation
      : capacity.recommendation;
  $("#text-bytes").textContent=`${capacity.text.utf8_bytes} B`;
  $("#safe-bytes").textContent=`${selected.payload.safe_text_bytes} B`;
  if(nmns){
    $("#vote-count").textContent=`${data.carrier_stats.used_branches}/${data.carrier_stats.branches}`;
    $("#vote-count-label").textContent="Branch lanes";
  }else{
    $("#vote-count").textContent=`${Number(selected.payload.mean_votes_per_bit).toFixed(1)}×`;
    $("#vote-count-label").textContent="Redundancy";
  }
  $("#resolution-grid").innerHTML=capacity.resolutions.map((row)=>`
    <button type="button" data-resolution="${row.resolution}" ${row.resource_limited?"disabled":""} class="${row.exportable?"ready":""} ${row.resource_limited?"limited":""} ${row.resolution===selected.resolution?"selected":""}">
      ${row.resolution}<br>${row.resource_limited?"Server limit":row.exportable?"Ready":"Full"}
    </button>`).join("");
  $("#resolution-grid").querySelectorAll("button").forEach((button)=>button.addEventListener("click",()=>{
    resolutionSelect.value=button.dataset.resolution;checkCapacity();
  }));
  if(ready){
    $("#export-panel").hidden=false;
    $("#export-panel").classList.remove("locked");
    $("#encoded-preview").src=`${data.preview_url}?v=${Date.now()}`;
    exportButton.disabled=false;
    $("#export-status").textContent=nmns?`${selected.resolution}px · vein graph PNG`:`${selected.resolution}px · JPG quality 96`;
    $("#sound-panel").hidden=nmns;
    $("#sound-panel").classList.toggle("locked",nmns);
    applySoundButton.disabled=nmns||!state.audio;
    if(!nmns)requestAnimationFrame(()=>requestAnimationFrame(drawSoundWave));
    setStep(4);
  }else{
    $("#export-panel").hidden=true;
    $("#sound-panel").hidden=true;
    exportButton.disabled=true;
    applySoundButton.disabled=true;
    setStep(3);
  }
}

async function checkCapacity(){
  const text=textInput.value.trim();
  if(!state.specimen)return;
  if(!text){textInput.focus();return;}
  capacityButton.disabled=true;exportButton.disabled=true;
  $("#capacity-result").hidden=true;
  setProgress("capacity",2,"Preparing text");
  try{
    const queued=await jsonRequest("/api/jobs/carrier",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({pipeline:state.sourceHint,sample:state.specimen.id,text,target_resolution:Number(resolutionSelect.value)}),
    });
    renderCapacity(await pollJob(queued.job_id,"capacity"));
  }catch(error){
    setProgress("capacity",100,"Could not calculate");
    $("#capacity-result").hidden=false;
    $("#capacity-badge").textContent="ERROR";
    $("#capacity-badge").classList.add("fail");
    $("#capacity-title").textContent="Capacity check failed";
    $("#capacity-note").textContent=error.message;
  }finally{capacityButton.disabled=false;}
}

async function exportCarrier(){
  if(!state.document?.exportable)return;
  const nmns=state.sourceHint==="nmns";
  exportButton.disabled=true;
  $("#export-status").textContent="Preparing download…";
  try{
    const response=await fetch(nmns?"/api/language/document/export":"/api/language/colour/export",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify(nmns?{packet:state.document.packet}:{sample:state.document.sample,text:state.document.text,target_resolution:state.document.selected.resolution}),
    });
    if(!response.ok){const error=await response.json().catch(()=>({}));throw new Error(error.error||"Export failed");}
    const blob=await response.blob();
    const disposition=response.headers.get("Content-Disposition")||"";
    const filename=disposition.match(/filename="([^"]+)"/)?.[1]||`leaf-carrier.${nmns?"png":"jpg"}`;
    const url=URL.createObjectURL(blob);
    const link=document.createElement("a");
    link.href=url;link.download=filename;document.body.append(link);link.click();link.remove();
    setTimeout(()=>URL.revokeObjectURL(url),30000);
    $("#export-status").textContent=`Downloaded · ${(blob.size/1024/1024).toFixed(1)} MB`;
  }catch(error){$("#export-status").textContent=error.message;}
  finally{exportButton.disabled=!state.document?.exportable;}
}

function soundConfig(){
  return {
    semantic_mode:$("#semantic-mode").value,
    semantic_loss:Number($("#semantic-loss").value)/100,
    strength:Number($("#sound-strength").value)/100,
    shift:Number($("#sound-shift").value)/100,
    erasure:Number($("#sound-erasure").value)/100,
    flip:Number($("#sound-flip").value)/100,
    clustering:Number($("#sound-clustering").value)/100,
  };
}

function showAudioMonitor(view){
  if(!state.audioResult)return;
  state.audioView=view;
  document.querySelectorAll("[data-monitor]").forEach((button)=>button.classList.toggle("active",button.dataset.monitor===view));
  const url=state.audioResult.output_urls?.[view];
  if(url)$("#sound-monitor-image").src=`${url}?v=${Date.now()}`;
}

function renderAudioResult(result){
  state.audioResult=result;
  $("#sound-result").hidden=false;
  $("#sound-carriers").textContent=Number(result.carrier.recognized_cells).toLocaleString();
  $("#sound-changed").textContent=Number(result.carrier.changed_cells).toLocaleString();
  $("#sound-confidence").textContent=`${Number(result.carrier.mean_confidence).toFixed(1)}%`;
  $("#sound-fidelity").textContent=`${Number(result.semantic?.fidelity??100).toFixed(1)}%`;
  const valid=Boolean(result.disturbed_decode?.valid);
  $("#sound-crc").textContent=valid?"VALID":"FAILED";
  $("#sound-crc").style.color=valid?"var(--green)":"var(--red)";
  const semantic=result.semantic||{};
  $("#sound-decode-status").textContent=valid
    ? `TRANSPORT VALID · ${semantic.source_altered?"SOURCE ALTERED":"SOURCE PRESERVED"} · ${semantic.affected_words||0} words affected${result.resolution?.upgraded?` · auto ${result.resolution.used}px`:""}`
    : result.decode_error||"PNG read completed, but CRC could not recover the complete text.";
  const recovered=result.disturbed_decode?.text||"";
  $("#sound-original-text").value=semantic.source_text||state.document?.text||"";
  $("#sound-original-text").hidden=false;
  $("#sound-decoded-text").value=recovered;
  $("#sound-decoded-text").hidden=!recovered;
  showAudioMonitor("disturbed");
  setStep(5);
}

async function applySound(){
  if(!state.audio||!state.document?.exportable||state.sourceHint!=="rubbing")return;
  resetAudioResult();
  applySoundButton.disabled=true;
  $("#audio-progress-bar").value=0;
  setProgress("audio",2,"Preparing carrier");
  try{
    const levels=16;
    const queued=await jsonRequest("/api/jobs/audio-disturbance",{
      method:"POST",headers:{"Content-Type":"application/json"},
      body:JSON.stringify({
        sample:state.document.sample,
        text:state.document.text,
        target_resolution:state.document.selected.resolution,
        levels,
        audio_array:quantizedAudio(levels),
        audio:{name:state.audio.name,duration:state.audio.duration,impacts:state.audio.impacts},
        config:soundConfig(),
      }),
    });
    renderAudioResult(await pollJob(queued.job_id,"audio"));
  }catch(error){
    setProgress("audio",100,"Could not apply sound");
    $("#sound-summary").textContent=error.message;
  }finally{
    applySoundButton.disabled=!state.audio||!state.document?.exportable||state.sourceHint!=="rubbing";
  }
}

async function downloadOutput(url,filename){
  if(!url)return;
  const response=await fetch(url);
  if(!response.ok)throw new Error("Download failed");
  const objectUrl=URL.createObjectURL(await response.blob());
  const link=document.createElement("a");
  link.href=objectUrl;link.download=filename;document.body.append(link);link.click();link.remove();
  setTimeout(()=>URL.revokeObjectURL(objectUrl),30000);
}

async function decodeImage(){
  const file=$("#decoder-file").files[0];
  if(!file){$("#decoder-status").textContent="Choose an exported JPG or PNG.";return;}
  const png=file.type==="image/png"||file.name.toLowerCase().endsWith(".png");
  $("#decode-image").disabled=true;
  $("#decoder-status").textContent="Reading image carrier…";
  try{
    const data=await fileDataUrl(file);
    const request={method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filename:file.name,data})};
    let result;
    try{
      result=await jsonRequest("/api/language/colour/decode",request);
    }catch(colourError){
      if(!png)throw colourError;
      $("#decoder-status").textContent="Matching NMNS vein graph…";
      result=await jsonRequest("/api/language/document/decode-png",request);
    }
    $("#decoder-status").textContent=result.valid?`Valid · text recovered${result.specimen?` · ${result.specimen}`:""}`:"Could not verify this image";
    $("#decoded-text").value=result.text||"";
    $("#decoded-text").hidden=!result.text;
  }catch(error){$("#decoder-status").textContent=error.message;}
  finally{$("#decode-image").disabled=false;}
}

document.querySelectorAll("[data-source]").forEach((button)=>button.addEventListener("click",()=>choosePipeline(button.dataset.source)));
document.querySelectorAll("[data-analysis-view]").forEach((button)=>button.addEventListener("click",()=>showAnalysisView(button.dataset.analysisView)));
[
  ["analysis-resolution","analysis-resolution-value"," px"],
  ["analysis-sensitivity","analysis-sensitivity-value","%"],
  ["analysis-contrast","analysis-contrast-value"," px"],
  ["analysis-component","analysis-component-value"," px"],
].forEach(([inputId,outputId,suffix])=>$("#"+inputId).addEventListener("input",()=>{$("#"+outputId).textContent=`${$("#"+inputId).value}${suffix}`;}));
$("#reanalyze-leaf").addEventListener("click",reanalyzeLeaf);
leafFile.addEventListener("change",()=>{$("#file-label").textContent=leafFile.files[0]?.name||"Choose an image";});
leafSelect.addEventListener("change",()=>{const specimen=state.specimens.find((item)=>item.id===leafSelect.value);if(specimen)renderLeaf(specimen);});
textInput.addEventListener("input",()=>{$("#text-count").textContent=`${new TextEncoder().encode(textInput.value).length} bytes`;resetDecision();});
resolutionSelect.addEventListener("change",resetDecision);
analyzeLeafButton.addEventListener("click",uploadLeaf);
capacityButton.addEventListener("click",checkCapacity);
exportButton.addEventListener("click",exportCarrier);
$("#decode-image").addEventListener("click",decodeImage);
soundFile.addEventListener("change",async()=>{
  const file=soundFile.files[0];
  if(!file)return;
  soundLibrary.value="";
  if(!file.name.toLowerCase().endsWith(".wav")){$("#sound-summary").textContent="Choose a WAV file.";return;}
  if(file.size>64*1024*1024){$("#sound-summary").textContent="The WAV is over 64 MB.";return;}
  setSoundPreview(file,file.name);
  await analyzeWav(file);
});
soundLibrary.addEventListener("change",()=>{
  if(soundLibrary.value!=="")analyzeLibraryAudio(soundLibrary.value);
  else clearSoundPreview();
});
applySoundButton.addEventListener("click",applySound);
document.querySelectorAll("[data-monitor]").forEach((button)=>button.addEventListener("click",()=>showAudioMonitor(button.dataset.monitor)));
[
  ["sound-strength","sound-strength-value"],
  ["sound-clustering","sound-clustering-value"],
  ["semantic-loss","semantic-loss-value"],
  ["sound-shift","sound-shift-value"],
  ["sound-erasure","sound-erasure-value"],
  ["sound-flip","sound-flip-value"],
].forEach(([inputId,valueId])=>{
  $("#"+inputId).addEventListener("input",()=>{
    $("#"+valueId).textContent=`${$("#"+inputId).value}%`;
    if(state.audioResult){resetAudioResult();setStep(3);}
  });
});
$("#semantic-mode").addEventListener("change",()=>{
  const defaults={preserve:0,trace:6,breakdown:60};
  $("#semantic-loss").value=defaults[$("#semantic-mode").value];
  $("#semantic-loss-value").textContent=`${$("#semantic-loss").value}%`;
  if(state.audioResult){resetAudioResult();setStep(3);}
});
$("#export-disturbed").addEventListener("click",()=>{
  const sample=state.document?.sample||"leaf";
  downloadOutput(state.audioResult?.output_urls?.disturbed,`${sample}_audio_disturbed.png`).catch((error)=>{$("#sound-decode-status").textContent=error.message;});
});
$("#export-audio-report").addEventListener("click",()=>{
  const sample=state.document?.sample||"leaf";
  downloadOutput(state.audioResult?.experiment_url,`${sample}_audio_experiment.json`).catch((error)=>{$("#sound-decode-status").textContent=error.message;});
});
window.addEventListener("resize",()=>drawSoundWave());
window.addEventListener("beforeunload",()=>{
  if(state.audioPreviewObjectUrl)URL.revokeObjectURL(state.audioPreviewObjectUrl);
});

textInput.disabled=true;resolutionSelect.disabled=true;capacityButton.disabled=true;
drawSoundWave();
loadAudioLibrary().catch((error)=>{
  soundLibrary.replaceChildren(new Option("Archive recordings unavailable", ""));
  soundLibrary.disabled=true;
  $("#sound-summary").textContent=error.message;
});
loadLibraries().catch((error)=>{$("#library-count").textContent=error.message;});
if(new URLSearchParams(location.search).get("panel")==="analysis")setTimeout(()=>$("#analysis-panel").scrollIntoView({behavior:"smooth"}),900);
