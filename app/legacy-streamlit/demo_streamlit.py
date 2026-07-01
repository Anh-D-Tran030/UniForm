from pathlib import Path
import json
from uuid import uuid4

import requests
import streamlit as st


DEFAULT_API_URL = "http://localhost:8005"
DEFAULT_KV_API_URL = "http://localhost:8006"
DEFAULT_STORAGE_API_URL = "http://localhost:8007"


def api_url():
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")


def kv_api_url():
    return st.session_state.get("kv_api_url", DEFAULT_KV_API_URL).rstrip("/")


def storage_api_url():
    return st.session_state.get("storage_api_url", DEFAULT_STORAGE_API_URL).rstrip("/")


def show_response_error(response):
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"API error {response.status_code}: {detail}")


def post_file(endpoint, uploaded_file, data=None, params=None):
    files = {
        "image": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }
    response = requests.post(
        f"{api_url()}{endpoint}",
        data=data,
        params=params,
        files=files,
        timeout=300,
    )
    if not response.ok:
        show_response_error(response)
        return None
    return response.json()


def post_file_bytes(base_url, endpoint, file_name, file_bytes, mime_type):
    files = {
        "image": (
            file_name,
            file_bytes,
            mime_type,
        )
    }
    response = requests.post(
        f"{base_url.rstrip('/')}{endpoint}",
        files=files,
        timeout=300,
    )
    return response


def guess_mime_type(path):
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def show_image_from_path(image_path, caption=None):
    if not image_path:
        return

    path = Path(image_path)
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.caption(image_path)


def new_run_id():
    return uuid4().hex


def build_kvp_review_payload(kvp_result, run_id, template_id, source_file_name):
    return {
        "run_id": run_id,
        "template_id": template_id,
        "source_file_name": source_file_name,
        "key_values": kvp_result.get("key_values", []),
    }


st.set_page_config(page_title="Uniform Demo", layout="centered")
st.title("UniForm - Unified Form Ingestion System")

if "extract_run_id" not in st.session_state:
    st.session_state["extract_run_id"] = new_run_id()
if "extract_template_id" not in st.session_state:
    st.session_state["extract_template_id"] = "unknown"
if "extract_source_path" not in st.session_state:
    st.session_state["extract_source_path"] = None
if "extract_source_name" not in st.session_state:
    st.session_state["extract_source_name"] = None
if "extract_overlay_bytes" not in st.session_state:
    st.session_state["extract_overlay_bytes"] = None
if "extract_overlay_path" not in st.session_state:
    st.session_state["extract_overlay_path"] = None
if "extract_uploaded_bytes" not in st.session_state:
    st.session_state["extract_uploaded_bytes"] = None
if "extract_uploaded_name" not in st.session_state:
    st.session_state["extract_uploaded_name"] = None
if "extract_uploaded_mime_type" not in st.session_state:
    st.session_state["extract_uploaded_mime_type"] = None
if "extract_kvp_json_text" not in st.session_state:
    st.session_state["extract_kvp_json_text"] = ""
if "extract_storage_result" not in st.session_state:
    st.session_state["extract_storage_result"] = None
if "query_uploaded_bytes" not in st.session_state:
    st.session_state["query_uploaded_bytes"] = None
if "query_uploaded_name" not in st.session_state:
    st.session_state["query_uploaded_name"] = None
if "query_uploaded_mime_type" not in st.session_state:
    st.session_state["query_uploaded_mime_type"] = None
if "query_result" not in st.session_state:
    st.session_state["query_result"] = None
tab_query, tab_add, tab_templates, tab_extract = st.tabs([
    "Query",
    "Add Template",
    "Templates",
    "Extract Key-Value",
])

with tab_query:
    st.subheader("Find matching templates")
    query_file = st.file_uploader("Query image", type=["png", "jpg", "jpeg"], key="query_file")
    top_k = st.number_input("Top matches", min_value=1, max_value=50, value=5, step=1)

    if query_file is not None:
        query_bytes = query_file.getvalue()
        query_changed = (
            query_bytes != st.session_state.get("query_uploaded_bytes")
            or query_file.name != st.session_state.get("query_uploaded_name")
        )
        st.session_state["query_uploaded_bytes"] = query_bytes
        st.session_state["query_uploaded_name"] = query_file.name
        st.session_state["query_uploaded_mime_type"] = query_file.type or "application/octet-stream"
        if query_changed:
            st.session_state["query_result"] = None

    if st.button("Search", type="primary", disabled=query_file is None):
        with st.spinner("Embedding query image and searching..."):
            result = post_file("/query", query_file, params={"top_k": int(top_k)})
        if result:
            st.session_state["query_result"] = result
            st.success(f"Found {len(result.get('matches', []))} match(es)")

    result = st.session_state.get("query_result")
    if result:
        show_image_from_path(result.get("query_image_path"), "Query image")

        for idx, match in enumerate(result.get("matches", [])):
            score = match.get("cosine_similarity", 0.0)
            title = match.get("display_name") or match.get("template_id")
            with st.container(border=True):
                cols = st.columns([1, 2, 1])
                with cols[0]:
                    show_image_from_path(match.get("image_path"), title)
                with cols[1]:
                    st.write(f"**{title}**")
                    st.write(f"Template ID: `{match.get('template_id')}`")
                    st.write(f"Similarity: `{score:.4f}`")
                    st.write(f"Word count: `{match.get('word_count')}`")
                with cols[2]:
                    if st.button("Use template for run", key=f"use_template_{idx}_{match.get('template_id')}"):
                        query_bytes = st.session_state.get("query_uploaded_bytes")
                        if query_bytes is None:
                            st.error("Query image is no longer available. Upload it again and search.")
                        else:
                            st.session_state["extract_source_path"] = None
                            st.session_state["extract_source_name"] = st.session_state.get("query_uploaded_name")
                            st.session_state["extract_template_id"] = match.get("template_id") or "unknown"
                            st.session_state["extract_uploaded_bytes"] = query_bytes
                            st.session_state["extract_uploaded_name"] = st.session_state.get("query_uploaded_name")
                            st.session_state["extract_uploaded_mime_type"] = (
                                st.session_state.get("query_uploaded_mime_type") or "application/octet-stream"
                            )
                            st.session_state["extract_overlay_bytes"] = None
                            st.session_state["extract_overlay_path"] = None
                            st.session_state["extract_kvp_json_text"] = ""
                            st.session_state["extract_storage_result"] = None
                            st.info(
                                f"Run template id updated to `{st.session_state['extract_template_id']}`. "
                                "Open the 'Extract Key-Value' tab when ready."
                            )

with tab_add:
    st.subheader("Add a template")
    template_id = st.text_input("Template ID")
    display_name = st.text_input("Display name")
    template_file = st.file_uploader("Template image", type=["png", "jpg", "jpeg"], key="template_file")

    can_add = bool(template_id.strip()) and template_file is not None
    if st.button("Embed Template", type="primary", disabled=not can_add):
        with st.spinner("Embedding and saving template..."):
            result = post_file(
                "/embed",
                template_file,
                data={
                    "template_id": template_id.strip(),
                    "display_name": display_name.strip(),
                },
            )

        if result:
            st.success(f"Saved template `{result.get('template_id')}`")
            show_image_from_path(result.get("image_path"), "Saved template")

with tab_templates:
    st.subheader("Saved templates")

    if st.button("Refresh Templates"):
        st.rerun()

    try:
        response = requests.get(f"{api_url()}/templates", timeout=60)
    except requests.RequestException as exc:
        st.error(f"Could not connect to backend: {exc}")
    else:
        if not response.ok:
            show_response_error(response)
        else:
            templates = response.json().get("templates", [])
            if not templates:
                st.info("No templates saved yet.")

            for template in templates:
                title = template.get("display_name") or template.get("template_id")
                with st.container(border=True):
                    cols = st.columns([1, 2, 1])
                    with cols[0]:
                        show_image_from_path(template.get("image_path"), title)
                    with cols[1]:
                        st.write(f"**{title}**")
                        st.write(f"Template ID: `{template.get('template_id')}`")
                        st.write(f"Word count: `{template.get('word_count')}`")
                    with cols[2]:
                        if st.button("Delete", key=f"delete_{template.get('id')}"):
                            delete_response = requests.delete(
                                f"{api_url()}/templates/{template.get('template_id')}",
                                timeout=60,
                            )
                            if delete_response.ok:
                                st.success("Deleted")
                                st.rerun()
                            else:
                                show_response_error(delete_response)

with tab_extract:
    st.subheader("Extract Key-Value")

    source_path = st.session_state.get("extract_source_path")
    source_name = st.session_state.get("extract_source_name") or "Selected template"

    run_cols = st.columns([2, 2, 1])
    with run_cols[2]:
        if st.button("Refresh run id"):
            st.session_state["extract_run_id"] = new_run_id()
            st.session_state["extract_storage_result"] = None
            if st.session_state.get("extract_kvp_json_text"):
                try:
                    current_payload = json.loads(st.session_state["extract_kvp_json_text"])
                except json.JSONDecodeError:
                    current_payload = None
                if isinstance(current_payload, dict):
                    current_payload["run_id"] = st.session_state["extract_run_id"]
                    current_payload["template_id"] = st.session_state.get("extract_template_id", "unknown")
                    st.session_state["extract_kvp_json_text"] = json.dumps(current_payload, indent=2)
            st.rerun()
    with run_cols[0]:
        st.text_input("Run ID", key="extract_run_id")
    with run_cols[1]:
        st.text_input("Template ID", key="extract_template_id")

    extract_file = st.file_uploader(
        "Upload image for extraction",
        type=["png", "jpg", "jpeg"],
        key="extract_file",
    )

    selected_bytes = None
    selected_file_name = None
    selected_mime_type = None

    if extract_file is not None:
        uploaded_bytes = extract_file.getvalue()
        upload_changed = (
            uploaded_bytes != st.session_state.get("extract_uploaded_bytes")
            or extract_file.name != st.session_state.get("extract_uploaded_name")
        )
        st.session_state["extract_uploaded_bytes"] = uploaded_bytes
        st.session_state["extract_uploaded_name"] = extract_file.name
        st.session_state["extract_uploaded_mime_type"] = extract_file.type or "application/octet-stream"
        if upload_changed:
            st.session_state["extract_source_path"] = None
            st.session_state["extract_source_name"] = None
            st.session_state["extract_overlay_bytes"] = None
            st.session_state["extract_overlay_path"] = None
            st.session_state["extract_kvp_json_text"] = ""
            st.session_state["extract_storage_result"] = None

    remembered_uploaded_bytes = st.session_state.get("extract_uploaded_bytes")
    remembered_uploaded_name = st.session_state.get("extract_uploaded_name")
    remembered_uploaded_mime_type = st.session_state.get("extract_uploaded_mime_type")

    if remembered_uploaded_bytes is not None:
        selected_bytes = remembered_uploaded_bytes
        selected_file_name = remembered_uploaded_name
        selected_mime_type = remembered_uploaded_mime_type or "application/octet-stream"
        st.image(selected_bytes, caption="Current uploaded file", use_column_width=True)
    elif source_path:
        path = Path(source_path)
        if path.exists():
            selected_bytes = path.read_bytes()
            selected_file_name = path.name
            selected_mime_type = guess_mime_type(path)
            st.image(str(path), caption=f"Current uploaded file: {source_name}", use_column_width=True)
        else:
            st.warning("Selected template image no longer exists. Upload a new file.")

    if selected_bytes is None:
        st.info("No current upload file. Please upload an image or choose Extract from query results.")

    if st.button("Clear remembered file", disabled=selected_bytes is None):
        st.session_state["extract_source_path"] = None
        st.session_state["extract_source_name"] = None
        st.session_state["extract_overlay_bytes"] = None
        st.session_state["extract_overlay_path"] = None
        st.session_state["extract_uploaded_bytes"] = None
        st.session_state["extract_uploaded_name"] = None
        st.session_state["extract_uploaded_mime_type"] = None
        st.session_state["extract_kvp_json_text"] = ""
        st.session_state["extract_storage_result"] = None
        st.rerun()

    if st.button("Extract", type="primary", disabled=selected_bytes is None):
        with st.spinner("Extracting key-value pairs..."):
            response = post_file_bytes(
                base_url=kv_api_url(),
                endpoint="/key-values",
                file_name=selected_file_name,
                file_bytes=selected_bytes,
                mime_type=selected_mime_type,
            )

        if response.ok:
            kvp_result = response.json()
            template_id_for_storage = st.session_state.get("extract_template_id", "unknown").strip() or "unknown"
            review_payload = build_kvp_review_payload(
                kvp_result=kvp_result,
                run_id=st.session_state["extract_run_id"],
                template_id=template_id_for_storage,
                source_file_name=selected_file_name,
            )
            st.session_state["extract_kvp_json_text"] = json.dumps(review_payload, indent=2)
            st.session_state["extract_overlay_path"] = None
            st.session_state["extract_overlay_bytes"] = None
            st.session_state["extract_storage_result"] = None
            with st.spinner("Rendering overlay preview..."):
                overlay_response = post_file_bytes(
                    base_url=kv_api_url(),
                    endpoint="/predict",
                    file_name=selected_file_name,
                    file_bytes=selected_bytes,
                    mime_type=selected_mime_type,
                )
            if overlay_response.ok:
                st.session_state["extract_overlay_bytes"] = overlay_response.content
                st.success("Extraction completed")
            else:
                st.warning("Key-value extraction completed, but overlay rendering failed.")
                show_response_error(overlay_response)
        else:
            st.session_state["extract_overlay_bytes"] = None
            st.session_state["extract_overlay_path"] = None
            show_response_error(response)

    st.text_area(
        "KVP JSON",
        key="extract_kvp_json_text",
        height=360,
        placeholder="Run extraction to populate editable KVP JSON before storing to MinIO.",
    )

    store_disabled = selected_bytes is None or not st.session_state.get("extract_kvp_json_text", "").strip()
    if st.button("Store to MinIO", type="primary", disabled=store_disabled):
        try:
            edited_kvp = json.loads(st.session_state["extract_kvp_json_text"])
        except json.JSONDecodeError as exc:
            st.error(f"Invalid JSON: {exc}")
        else:
            with st.spinner("Writing bronze image and silver JSON to MinIO..."):
                files = {
                    "image": (
                        selected_file_name or "upload.png",
                        selected_bytes,
                        selected_mime_type or "application/octet-stream",
                    )
                }
                data = {
                    "run_id": st.session_state["extract_run_id"].strip() or new_run_id(),
                    "template_id": st.session_state.get("extract_template_id", "unknown").strip() or "unknown",
                    "kvp_json": json.dumps(edited_kvp),
                }
                try:
                    store_response = requests.post(
                        f"{storage_api_url()}/ingest",
                        data=data,
                        files=files,
                        timeout=120,
                    )
                except requests.RequestException as exc:
                    st.error(f"Could not connect to MinIO storage service: {exc}")
                else:
                    if store_response.ok:
                        st.session_state["extract_storage_result"] = store_response.json()
                        st.success("Stored to MinIO")
                    else:
                        show_response_error(store_response)

    storage_result = st.session_state.get("extract_storage_result")
    if storage_result:
        st.write("Stored objects")
        st.json(storage_result)

    overlay_path = st.session_state.get("extract_overlay_path")
    if overlay_path:
        show_image_from_path(overlay_path, "Overlay key-value result")

    overlay_bytes = st.session_state.get("extract_overlay_bytes")
    if overlay_bytes:
        st.image(overlay_bytes, caption="Overlay key-value result", use_column_width=True)
