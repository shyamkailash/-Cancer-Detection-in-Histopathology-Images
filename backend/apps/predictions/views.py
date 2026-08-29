"""
REST API views for Histopathology Cancer Detection predictions and Grad-CAM explainability.
"""

from django.http import HttpResponse
from rest_framework import status
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import AllowAny
from rest_framework.parsers import MultiPartParser, FormParser, JSONParser

from ml.inference.pipeline import get_inference_pipeline
from ml.inference.model_manager import get_model_manager


class ModelsListAPIView(APIView):
    """
    GET /api/models/
    List all available cancer detection model architectures and their checkpoint status.
    """
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        manager = get_model_manager()
        models = manager.list_models()
        return Response({
            "status": "success",
            "count": len(models),
            "models": models,
        })


class PredictAPIView(APIView):
    """
    POST /api/predict/
    Upload a histopathology image patch (or base64 URI) to obtain cancer classification & Grad-CAM heatmap.
    """
    permission_classes = [AllowAny]
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request, *args, **kwargs):
        # 1. Retrieve image payload
        image_file = request.FILES.get("image")
        image_base64 = request.data.get("image_base64") or request.data.get("image")

        if not image_file and not image_base64:
            return Response(
                {
                    "status": "error",
                    "error": "No image provided. Please upload an image file via multipart/form-data 'image' field or provide base64 data in 'image_base64'.",
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        # 2. Retrieve parameters
        model_name = request.data.get("model_name") or request.data.get("model", "centralized")
        include_gradcam_raw = request.data.get("include_gradcam", True)
        if isinstance(include_gradcam_raw, str):
            include_gradcam = include_gradcam_raw.lower() in ("true", "1", "yes")
        else:
            include_gradcam = bool(include_gradcam_raw)

        target_class_raw = request.data.get("target_class")
        target_class = int(target_class_raw) if target_class_raw is not None and str(target_class_raw).isdigit() else None

        # 3. Execute inference pipeline
        pipeline = get_inference_pipeline()
        try:
            image_input = image_file.read() if image_file else image_base64
            result = pipeline.predict(
                image_input=image_input,
                model_name=model_name,
                include_gradcam=include_gradcam,
                target_class=target_class,
            )
            return Response(result, status=status.HTTP_200_OK)

        except ValueError as val_err:
            return Response(
                {"status": "error", "error": str(val_err)},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as exc:
            return Response(
                {
                    "status": "error",
                    "error": f"Inference execution failed: {str(exc)}",
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


class DemoView(APIView):
    """
    GET /api/demo/
    Interactive presentation UI for live demonstration of Cancer Detection and Grad-CAM heatmaps.
    """
    permission_classes = [AllowAny]

    def get(self, request, *args, **kwargs):
        html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Site Cancer Detection & Explainability Demo</title>
    <style>
        :root {
            --primary: #2563eb;
            --primary-hover: #1d4ed8;
            --success: #16a34a;
            --danger: #dc2626;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
        body { background: var(--bg); color: var(--text-main); line-height: 1.5; padding: 24px; }
        .container { max-width: 1100px; margin: 0 auto; }
        header { text-align: center; margin-bottom: 28px; }
        h1 { font-size: 1.85rem; font-weight: 700; color: var(--text-main); margin-bottom: 8px; }
        .subtitle { color: var(--text-muted); font-size: 0.95rem; }
        .grid { display: grid; grid-template-columns: 360px 1fr; gap: 24px; }
        .card { background: var(--card-bg); border-radius: 12px; border: 1px solid var(--border); box-shadow: 0 1px 3px rgba(0,0,0,0.05); padding: 20px; }
        .card h2 { font-size: 1.15rem; font-weight: 600; margin-bottom: 16px; border-bottom: 1px solid var(--border); padding-bottom: 8px; }
        .dropzone { border: 2px dashed var(--border); border-radius: 8px; padding: 28px 16px; text-align: center; cursor: pointer; transition: all 0.2s; background: #fdfdfd; }
        .dropzone:hover { border-color: var(--primary); background: #eff6ff; }
        .form-group { margin-top: 16px; }
        label { display: block; font-size: 0.85rem; font-weight: 600; margin-bottom: 6px; color: var(--text-main); }
        select, button { width: 100%; padding: 10px 12px; border-radius: 6px; font-size: 0.9rem; }
        select { border: 1px solid var(--border); background: white; outline: none; }
        button { background: var(--primary); color: white; border: none; font-weight: 600; cursor: pointer; margin-top: 16px; transition: background 0.2s; }
        button:hover { background: var(--primary-hover); }
        .preview-box { margin-top: 12px; display: none; text-align: center; }
        .preview-box img { max-width: 140px; max-height: 140px; border-radius: 6px; border: 1px solid var(--border); }
        .results-section { display: none; }
        .pred-badge { display: inline-block; padding: 6px 14px; border-radius: 20px; font-weight: 700; font-size: 1rem; margin-bottom: 12px; }
        .badge-normal { background: #dcfce7; color: var(--success); }
        .badge-metastasis { background: #fee2e2; color: var(--danger); }
        .prob-bar { background: #e2e8f0; border-radius: 4px; height: 10px; overflow: hidden; margin: 4px 0 12px 0; }
        .prob-fill { height: 100%; transition: width 0.4s ease; }
        .fill-normal { background: var(--success); }
        .fill-metastasis { background: var(--danger); }
        .img-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-top: 20px; }
        .img-card { text-align: center; background: #fafafa; border: 1px solid var(--border); border-radius: 8px; padding: 10px; }
        .img-card img { width: 100%; height: auto; border-radius: 4px; object-fit: cover; }
        .img-card span { display: block; font-size: 0.8rem; font-weight: 600; margin-top: 6px; color: var(--text-muted); }
        .meta-tag { font-size: 0.8rem; color: var(--text-muted); margin-top: 10px; }
        .disclaimer-box { margin-top: 24px; padding: 12px 16px; background: #fffbeb; border: 1px solid #fef3c7; border-radius: 8px; font-size: 0.8rem; color: #92400e; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Privacy-Preserving Federated Cancer Detection</h1>
            <p class="subtitle">Multi-Site Histopathology Patch Classification & Grad-CAM Visual Explainability</p>
        </header>

        <div class="grid">
            <!-- Upload Column -->
            <div class="card">
                <h2>1. Select Histopathology Patch</h2>
                <div class="dropzone" id="dropzone" onclick="document.getElementById('fileInput').click()">
                    <p style="font-size: 0.9rem; color: var(--text-muted);">Click or Drag & Drop Image Patch</p>
                    <p style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">PNG, JPG, TIFF (96x96px)</p>
                </div>
                <input type="file" id="fileInput" accept="image/*" style="display:none" onchange="handleFile(this.files[0])">
                
                <div class="preview-box" id="previewBox">
                    <img id="uploadPreview" src="" alt="Patch Preview">
                </div>

                <div class="form-group">
                    <label for="modelSelect">2. Choose Model Architecture</label>
                    <select id="modelSelect">
                        <option value="centralized">Centralized ResNet-18 (94.13% Acc)</option>
                        <option value="fedavg">Federated ResNet-18 FedAvg (88.80% Acc)</option>
                        <option value="dp_fedavg">Privacy-Preserving DP-FedAvg (67.20% Acc)</option>
                    </select>
                </div>

                <button id="predictBtn" onclick="runPrediction()">Analyze Patch</button>
            </div>

            <!-- Results Column -->
            <div class="card">
                <h2>3. Clinical Prediction & Grad-CAM Explainability</h2>
                <div id="loading" style="display:none; text-align:center; padding: 40px;">
                    <p style="color: var(--primary); font-weight: 600;">Running Inference & Backpropagating Grad-CAM...</p>
                </div>

                <div id="resultsSection" class="results-section">
                    <div id="predBadge" class="pred-badge"></div>
                    <div style="font-size: 0.85rem; color: var(--text-muted); margin-bottom: 12px;" id="predDesc"></div>

                    <div>
                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; font-weight:600;">
                            <span>Metastasis Probability (Tumor)</span>
                            <span id="probMetastasisText">0%</span>
                        </div>
                        <div class="prob-bar"><div id="fillMetastasis" class="prob-fill fill-metastasis" style="width:0%"></div></div>

                        <div style="display:flex; justify-content:space-between; font-size:0.8rem; font-weight:600;">
                            <span>Normal Tissue Probability</span>
                            <span id="probNormalText">0%</span>
                        </div>
                        <div class="prob-bar"><div id="fillNormal" class="prob-fill fill-normal" style="width:0%"></div></div>
                    </div>

                    <div class="img-grid">
                        <div class="img-card">
                            <img id="imgOriginal" src="" alt="Original Patch">
                            <span>Input Patch</span>
                        </div>
                        <div class="img-card">
                            <img id="imgHeatmap" src="" alt="Grad-CAM Heatmap">
                            <span>Grad-CAM Heatmap</span>
                        </div>
                        <div class="img-card">
                            <img id="imgOverlay" src="" alt="Grad-CAM Overlay">
                            <span>Diagnostic Overlay</span>
                        </div>
                    </div>

                    <div class="meta-tag" id="metaInfo"></div>
                </div>

                <div id="emptyState" style="text-align:center; padding: 60px 20px; color: var(--text-muted);">
                    <p>Upload an image patch to visualize cancer prediction and Grad-CAM activation overlays.</p>
                </div>
            </div>
        </div>

        <div class="disclaimer-box">
            <strong>Clinical Disclaimer:</strong> This system is an academic research and educational prototype. It is not certified for clinical diagnosis, patient triage, or healthcare decisions.
        </div>
    </div>

    <script>
        let selectedFile = null;

        function handleFile(file) {
            if (!file) return;
            selectedFile = file;
            const reader = new FileReader();
            reader.onload = e => {
                document.getElementById('uploadPreview').src = e.target.result;
                document.getElementById('previewBox').style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        const dropzone = document.getElementById('dropzone');
        dropzone.addEventListener('dragover', e => { e.preventDefault(); dropzone.style.borderColor = '#2563eb'; });
        dropzone.addEventListener('dragleave', e => { dropzone.style.borderColor = '#e2e8f0'; });
        dropzone.addEventListener('drop', e => {
            e.preventDefault();
            dropzone.style.borderColor = '#e2e8f0';
            if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
        });

        async function runPrediction() {
            if (!selectedFile) {
                alert("Please select or drop an image patch first.");
                return;
            }
            const modelName = document.getElementById('modelSelect').value;
            const formData = new FormData();
            formData.append('image', selectedFile);
            formData.append('model_name', modelName);
            formData.append('include_gradcam', 'true');

            document.getElementById('emptyState').style.display = 'none';
            document.getElementById('resultsSection').style.display = 'none';
            document.getElementById('loading').style.display = 'block';

            try {
                const resp = await fetch('/api/predict/', { method: 'POST', body: formData });
                const data = await resp.json();
                document.getElementById('loading').style.display = 'none';

                if (data.status === 'success') {
                    renderResults(data);
                } else {
                    alert("Prediction error: " + (data.error || "Unknown error"));
                }
            } catch (err) {
                document.getElementById('loading').style.display = 'none';
                alert("Network error: " + err.message);
            }
        }

        function renderResults(data) {
            document.getElementById('resultsSection').style.display = 'block';
            const pred = data.prediction;
            const badge = document.getElementById('predBadge');
            if (pred.class_id === 1) {
                badge.className = 'pred-badge badge-metastasis';
                badge.innerText = 'Metastatic Tumor Detected (' + (pred.confidence * 100).toFixed(1) + '%)';
            } else {
                badge.className = 'pred-badge badge-normal';
                badge.innerText = 'Normal Tissue - No Tumor (' + (pred.confidence * 100).toFixed(1) + '%)';
            }
            document.getElementById('predDesc').innerText = pred.description;

            const pMeta = (pred.probabilities.metastasis * 100).toFixed(1);
            const pNorm = (pred.probabilities.normal * 100).toFixed(1);
            document.getElementById('probMetastasisText').innerText = pMeta + '%';
            document.getElementById('fillMetastasis').style.width = pMeta + '%';
            document.getElementById('probNormalText').innerText = pNorm + '%';
            document.getElementById('fillNormal').style.width = pNorm + '%';

            if (data.explainability) {
                document.getElementById('imgOriginal').src = data.explainability.original_base64;
                document.getElementById('imgHeatmap').src = data.explainability.heatmap_base64;
                document.getElementById('imgOverlay').src = data.explainability.overlay_base64;
            }

            document.getElementById('metaInfo').innerText =
                'Model: ' + data.model.display_name + ' | Paradigm: ' + data.model.paradigm + ' | Inference: ' + data.inference_time_ms + 'ms | Device: ' + data.model.device;
        }
    </script>
</body>
</html>"""
        return HttpResponse(html_content, content_type="text/html")
