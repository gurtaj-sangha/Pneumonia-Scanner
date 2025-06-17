
document.getElementById("scanBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("xrayInput");
  if (!fileInput.files.length) {
    alert("Please choose an X-ray image first.");
    return;
  }

  const formData = new FormData();
  formData.append("xray_image", fileInput.files[0]);

  try {
    const res = await fetch("/predict", { method: "POST", body: formData });
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();

    document.getElementById("resultImg").src = data.image_url;
    document.getElementById("heatImg").src   = data.vis_url;

    
    const box = document.getElementById("probBox");
    box.innerHTML = ""; 
    Object.entries(data.probs).forEach(([label, p]) => {
      const line = document.createElement("div");
      line.textContent = `${label}: ${(p * 100).toFixed(1)} %`;
      box.appendChild(line);
    });
  } catch (err) {
    console.error(err);
    alert("Error while scanning image — check console.");
  }
});

