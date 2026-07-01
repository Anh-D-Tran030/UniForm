from pathlib import Path
import requests
import streamlit as st

st.set_page_config(page_title="Uniforn", layout="wide")
st.title("UniForm - Unified Form Ingestion System")

DEFAULT_API_URL = "http://localhost:8005"
DEFAULT_KV_API_URL = "http://localhost:8006"


def api_url() -> str:
    return st.session_state.get("api_url", DEFAULT_API_URL).rstrip("/")

def show_response_error(response: requests.Response) -> None:
    try:
        detail = response.json().get("detail", response.text)
    except ValueError:
        detail = response.text
    st.error(f"API error {response.status_code}: {detail}")

def post_file(endpoint, uploaded_file, data= None, params=None):
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

def show_image_from_path(image_path: str | None, caption: str | None = None) -> None:
    if not image_path:
        return

    path = Path(image_path)
    if path.exists():
        st.image(str(path), caption=caption, use_column_width=True)
    else:
        st.caption(image_path)

tab_query, tab_add, tab_templates, tab_extract = st.tabs([
    "Query",
    "Add Template",
    "Templates",
    "Extract Key-Value",
])
with tab_query:
    st.subheader("Upload to find matching template")
    query_file = st.file_uploader("Query image", type=["png", "jpg", "jpeg"], key="query_file")
    top_k = st.number_input("Top matches", min_value=1, max_value=50, value=5, step=1)
    if st.button("Search", type="primary", disabled=query_file is None):
        with st.spinner("Embedding query image and searching..."):
            result = post_file("/query", query_file, params={"top_k": int(top_k)})
        if result:
            st.success(f"Found {len(result.get('matches', []))} match(es)")
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
                    with cols[2]:
                        if st.button("Extract", key=f"extract_match_{idx}_{match.get('template_id')}"):
                            st.session_state["extract_source_path"] = match.get("image_path")
                            st.session_state["extract_source_name"] = title
                            st.session_state["extract_overlay_bytes"] = None
                            st.info("Source captured. Open the 'Extract Key-Value' tab.")
        
with tab_extract:
    st.subheader("Extract Key-Value")
with tab_add:
    
    st.subheader("Add new template")

    
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
    st.subheader("Available template")
    if st.button("Refresh Templates"):
        st.rerun()
    response = requests.get(f"{api_url()}/templates", timeout=60)
    if not response.ok:
        show_response_error(response)
    else:
        templates = response.json().get("templates",[])
        if not templates:
            st.info("There is no template")
        for template in templates:
            title = template.get("display_name") or template.get("template_id")
            with st.container(border=True):
                cols = st.columns([1, 2, 1])
                with cols[0]:
                    show_image_from_path(template.get("image_path"), title)
                with cols[1]:
                    st.write(f"**{title}**")
                    st.write(f"Template ID: `{template.get('template_id')}`")
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
