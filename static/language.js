const $=(selector)=>document.querySelector(selector);

const state={
  sourceHint:"rubbing",
  libraries:{rubbing:[],nmns:[]},
  specimens:[],
  specimen:null,
  document:null,
};

const leafFile=$("#leaf-file");
const leafSelect=$("#leaf-select");
const textInput=$("#text-input");
const resolutionSelect=$("#resolution-select");
const analyzeLeafButton=$("#analyze-leaf");
const capacityButton=$("#check-capacity");
const exportButton=$("#export-jpg");

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
  $("#capacity-result").hidden=true;
  $("#capacity-progress").hidden=true;
  $("#capacity-progress-bar").value=0;
  $("#export-panel").classList.add("locked");
  $("#export-panel").hidden=true;
  $("#encoded-preview").removeAttribute("src");
  exportButton.disabled=true;
  $("#export-status").textContent="Run the capacity check first.";
  setStep(state.specimen?2:1);
}

function setResolutionOptions(){
  const values=state.sourceHint==="nmns"?[2048,4096,8192,16384]:[2048,3072,4096,6144,8192];
  const preferred=state.sourceHint==="nmns"?8192:4096;
  resolutionSelect.innerHTML=values.map((value)=>`<option value="${value}" ${value===preferred?"selected":""}>${value} px</option>`).join("");
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
  $("#decoder-status").textContent="";
}

async function loadLibraries(){
  const [rubbing,nmns]=await Promise.all([
    jsonRequest("/api/colour/specimens"),
    jsonRequest("/api/images"),
  ]);
  state.libraries.rubbing=(rubbing.specimens||[]).map((item)=>({...item,pipeline:"rubbing"}));
  state.libraries.nmns=(nmns.images||[]).filter((item)=>item.ready).map((item)=>({
    ...item,
    pipeline:"nmns",
    id:item.id||item.name,
    name:item.display_name||item.name,
    source_url:item.source_url||`/source/${encodeURIComponent(item.name)}`,
  }));
  choosePipeline(state.sourceHint);
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
  $("#capacity-badge").textContent=ready?"READY":"TOO LARGE";
  $("#capacity-badge").classList.toggle("fail",!ready);
  $("#capacity-title").textContent=ready?"Ready to export":"More space needed";
  $("#capacity-note").textContent=ready?`${selected.resolution}px safely carries this text.`:capacity.recommendation;
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
    <button type="button" data-resolution="${row.resolution}" class="${row.exportable?"ready":""} ${row.resolution===selected.resolution?"selected":""}">
      ${row.resolution}<br>${row.exportable?"Ready":"Full"}
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
    setStep(3);
  }else{
    $("#export-panel").hidden=true;
    exportButton.disabled=true;
    setStep(2);
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

async function decodeImage(){
  const file=$("#decoder-file").files[0];
  if(!file){$("#decoder-status").textContent="Choose an exported JPG or PNG.";return;}
  const png=file.type==="image/png"||file.name.toLowerCase().endsWith(".png");
  $("#decode-image").disabled=true;
  $("#decoder-status").textContent=png?"Matching vein graph…":"Reading pixel carrier…";
  try{
    const data=await fileDataUrl(file);
    const result=await jsonRequest(png?"/api/language/document/decode-png":"/api/language/colour/decode",{
      method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({filename:file.name,data}),
    });
    $("#decoder-status").textContent=result.valid?`Valid · text recovered${result.specimen?` · ${result.specimen}`:""}`:"Could not verify this image";
    $("#decoded-text").value=result.text||"";
    $("#decoded-text").hidden=!result.text;
  }catch(error){$("#decoder-status").textContent=error.message;}
  finally{$("#decode-image").disabled=false;}
}

document.querySelectorAll("[data-source]").forEach((button)=>button.addEventListener("click",()=>choosePipeline(button.dataset.source)));
leafFile.addEventListener("change",()=>{$("#file-label").textContent=leafFile.files[0]?.name||"Choose an image";});
leafSelect.addEventListener("change",()=>{const specimen=state.specimens.find((item)=>item.id===leafSelect.value);if(specimen)renderLeaf(specimen);});
textInput.addEventListener("input",()=>{$("#text-count").textContent=`${new TextEncoder().encode(textInput.value).length} bytes`;resetDecision();});
resolutionSelect.addEventListener("change",resetDecision);
analyzeLeafButton.addEventListener("click",uploadLeaf);
capacityButton.addEventListener("click",checkCapacity);
exportButton.addEventListener("click",exportCarrier);
$("#decode-image").addEventListener("click",decodeImage);

textInput.disabled=true;resolutionSelect.disabled=true;capacityButton.disabled=true;
loadLibraries().catch((error)=>{$("#library-count").textContent=error.message;});
