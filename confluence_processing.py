# confluence_processing.py
"""
Module pour l'extraction de contenu depuis Confluence.
Supporte Confluence Cloud (atlassian.net) et Confluence Server.
"""

import re
import logging
import requests
from typing import List, Dict, Any, Optional, Callable
from html import unescape

# Pour le parsing HTML
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    logging.warning("BeautifulSoup4 non disponible. Installez: pip install beautifulsoup4")


ProgressFn = Callable[[float, str], None]


def _clean_html_to_text(html_content: str) -> str:
    """
    Convertit du contenu HTML en texte propre.
    Utilise BeautifulSoup si disponible, sinon regex basique.
    """
    if not html_content:
        return ""

    if BS4_AVAILABLE:
        soup = BeautifulSoup(html_content, "html.parser")

        # Supprimer les scripts et styles
        for element in soup(["script", "style", "nav", "footer", "header"]):
            element.decompose()

        # Remplacer les <br> et </p> par des sauts de ligne
        for br in soup.find_all("br"):
            br.replace_with("\n")
        for p in soup.find_all("p"):
            p.append("\n")

        # Traiter les listes
        for li in soup.find_all("li"):
            li.insert(0, "• ")
            li.append("\n")

        # Traiter les tableaux (format simple)
        for table in soup.find_all("table"):
            rows = table.find_all("tr")
            table_text = []
            for row in rows:
                cells = row.find_all(["td", "th"])
                row_text = " | ".join(cell.get_text(strip=True) for cell in cells)
                table_text.append(row_text)
            table.replace_with("\n".join(table_text) + "\n")

        # Traiter les headers
        for i in range(1, 7):
            for header in soup.find_all(f"h{i}"):
                header.insert(0, "\n" + "#" * i + " ")
                header.append("\n")

        # Extraire le texte
        text = soup.get_text()
    else:
        # Fallback sans BeautifulSoup
        text = html_content
        # Supprimer les tags HTML
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<br\s*/?>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</p>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)

    # Décoder les entités HTML
    text = unescape(text)

    # Normaliser les espaces et sauts de ligne
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join(line.strip() for line in text.split('\n'))

    return text.strip()


def _get_auth(username: str, password: str) -> tuple:
    """Retourne le tuple d'authentification pour requests."""
    return (username, password)


def _detect_confluence_type(base_url: str) -> str:
    """
    Détecte si c'est Confluence Cloud ou Server.
    Cloud: utilise /wiki/rest/api
    Server: utilise /rest/api
    """
    if "atlassian.net" in base_url.lower():
        return "cloud"
    return "server"


def _build_api_url(base_url: str, endpoint: str, context_path: str = "") -> str:
    """Construit l'URL de l'API en fonction du type de Confluence.

    Args:
        base_url: URL de base (ex: https://confluence.company.com)
        endpoint: Endpoint API (ex: user/current)
        context_path: Chemin de contexte optionnel (ex: /confluence, /wiki)
    """
    base_url = base_url.rstrip("/")

    # Si un contexte est spécifié, l'utiliser
    if context_path:
        context_path = context_path.strip("/")
        return f"{base_url}/{context_path}/rest/api/{endpoint}"

    conf_type = _detect_confluence_type(base_url)

    if conf_type == "cloud":
        # Confluence Cloud: base_url/wiki/rest/api/...
        if "/wiki" not in base_url:
            base_url = f"{base_url}/wiki"
        return f"{base_url}/rest/api/{endpoint}"
    else:
        # Confluence Server: essayer de détecter le contexte dans l'URL
        # Patterns courants: /confluence, /wiki, ou directement /rest/api
        if "/confluence" in base_url.lower() or "/wiki" in base_url.lower():
            return f"{base_url}/rest/api/{endpoint}"
        else:
            # Par défaut, pas de contexte
            return f"{base_url}/rest/api/{endpoint}"


def test_confluence_connection(
    base_url: str,
    username: str,
    password: str,
    verify_ssl: bool = True,
    context_path: str = "",
) -> Dict[str, Any]:
    """
    Teste la connexion à Confluence.

    Args:
        base_url: URL de base Confluence
        username: Nom d'utilisateur
        password: Mot de passe ou token API
        verify_ssl: Si False, désactive la vérification SSL (utile pour certificats auto-signés)
        context_path: Chemin de contexte pour Confluence Server (ex: /confluence, /wiki)

    Returns:
        Dict avec 'success', 'message', et optionnellement 'user_info'
    """
    # Désactiver les warnings SSL si verify_ssl=False
    if not verify_ssl:
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    try:
        api_url = _build_api_url(base_url, "user/current", context_path)
        logging.info(f"[confluence] Testing connection to: {api_url}")
        response = requests.get(
            api_url,
            auth=_get_auth(username, password),
            timeout=10,
            verify=verify_ssl,
        )

        logging.info(f"[confluence] Response status: {response.status_code}")
        logging.info(f"[confluence] Response content-type: {response.headers.get('content-type', 'unknown')}")

        if response.status_code == 200:
            # Vérifier que c'est bien du JSON
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type:
                return {
                    "success": False,
                    "message": f"Réponse non-JSON reçue (content-type: {content_type}). Le serveur renvoie probablement une page HTML. Vérifiez l'URL.",
                }
            try:
                user_info = response.json()
            except Exception as json_err:
                # Afficher un extrait de la réponse pour debug
                preview = response.text[:500] if response.text else "(vide)"
                return {
                    "success": False,
                    "message": f"Erreur JSON: {json_err}. Réponse reçue: {preview}",
                }
            return {
                "success": True,
                "message": f"Connecté en tant que: {user_info.get('displayName', username)}",
                "user_info": user_info,
            }
        elif response.status_code == 401:
            return {
                "success": False,
                "message": "Authentification échouée. Vérifiez vos identifiants.",
            }
        elif response.status_code == 404:
            return {
                "success": False,
                "message": f"API non trouvée à {api_url}. Vérifiez l'URL de base.",
            }
        else:
            return {
                "success": False,
                "message": f"Erreur HTTP {response.status_code}: {response.text[:200]}",
            }

    except requests.exceptions.SSLError as e:
        return {
            "success": False,
            "message": f"Erreur SSL: {str(e)}. Essayez de cocher 'Ignorer certificat SSL'.",
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "success": False,
            "message": f"Impossible de se connecter à {base_url}. Erreur: {str(e)[:100]}",
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Erreur: {str(e)}",
        }


def list_spaces(
    base_url: str,
    username: str,
    password: str,
    verify_ssl: bool = True,
    context_path: str = "",
) -> List[Dict[str, str]]:
    """
    Liste tous les espaces Confluence accessibles.

    Args:
        context_path: Chemin de contexte pour Confluence Server (ex: /confluence, /wiki)

    Returns:
        Liste de dicts avec 'key', 'name', 'type'
    """
    spaces = []
    start = 0
    limit = 50

    while True:
        api_url = _build_api_url(base_url, f"space?start={start}&limit={limit}", context_path)
        response = requests.get(
            api_url,
            auth=_get_auth(username, password),
            timeout=30,
            verify=verify_ssl,
        )

        if response.status_code != 200:
            logging.error(f"[confluence] Erreur listing spaces: {response.status_code}")
            break

        data = response.json()
        results = data.get("results", [])

        for space in results:
            spaces.append({
                "key": space.get("key", ""),
                "name": space.get("name", ""),
                "type": space.get("type", ""),
            })

        # Pagination
        if len(results) < limit:
            break
        start += limit

    return spaces


def get_space_pages(
    base_url: str,
    space_key: str,
    username: str,
    password: str,
    progress_cb: Optional[ProgressFn] = None,
    verify_ssl: bool = True,
    context_path: str = "",
) -> List[Dict[str, Any]]:
    """
    Récupère toutes les pages d'un espace Confluence.

    Args:
        context_path: Chemin de contexte pour Confluence Server (ex: /confluence, /wiki)

    Returns:
        Liste de dicts avec 'id', 'title', 'url', 'content' (HTML)
    """
    pages = []
    start = 0
    limit = 25  # Confluence limite souvent à 25 avec expand

    if progress_cb:
        progress_cb(0.0, f"Récupération des pages de l'espace {space_key}...")

    while True:
        api_url = _build_api_url(
            base_url,
            f"content?spaceKey={space_key}&type=page&start={start}&limit={limit}"
            f"&expand=body.storage,version,ancestors",
            context_path
        )

        try:
            response = requests.get(
                api_url,
                auth=_get_auth(username, password),
                timeout=60,
                verify=verify_ssl,
            )
        except requests.exceptions.Timeout:
            logging.error(f"[confluence] Timeout lors de la récupération des pages (start={start})")
            break

        if response.status_code != 200:
            logging.error(f"[confluence] Erreur {response.status_code}: {response.text[:200]}")
            break

        data = response.json()
        results = data.get("results", [])

        for page in results:
            page_id = page.get("id", "")
            title = page.get("title", "")

            # Contenu HTML
            body = page.get("body", {})
            storage = body.get("storage", {})
            html_content = storage.get("value", "")

            # URL de la page
            links = page.get("_links", {})
            web_ui = links.get("webui", "")
            base = links.get("base", base_url.rstrip("/"))
            page_url = f"{base}{web_ui}" if web_ui else ""

            # Ancêtres (pour le chemin hiérarchique)
            ancestors = page.get("ancestors", [])
            path = " > ".join([a.get("title", "") for a in ancestors])
            if path:
                path = f"{path} > {title}"
            else:
                path = title

            pages.append({
                "id": page_id,
                "title": title,
                "url": page_url,
                "path": path,
                "html_content": html_content,
            })

        if progress_cb:
            progress_cb(0.3, f"Pages récupérées: {len(pages)}...")

        # Pagination
        if len(results) < limit:
            break
        start += limit

    logging.info(f"[confluence] {len(pages)} pages récupérées depuis l'espace {space_key}")
    return pages


def extract_text_from_confluence_space(
    base_url: str,
    space_key: str,
    username: str,
    password: str,
    progress_cb: Optional[ProgressFn] = None,
    verify_ssl: bool = True,
    context_path: str = "",
) -> List[Dict[str, Any]]:
    """
    Extrait le texte de toutes les pages d'un espace Confluence.

    Args:
        context_path: Chemin de contexte pour Confluence Server (ex: /confluence, /wiki)

    Returns:
        Liste de dicts avec:
        - 'id': ID de la page
        - 'title': Titre de la page
        - 'url': URL de la page
        - 'path': Chemin hiérarchique
        - 'text': Contenu texte extrait
    """
    if progress_cb:
        progress_cb(0.0, "Connexion à Confluence...")

    # Récupérer les pages
    pages = get_space_pages(base_url, space_key, username, password, progress_cb, verify_ssl=verify_ssl, context_path=context_path)

    if not pages:
        logging.warning(f"[confluence] Aucune page trouvée dans l'espace {space_key}")
        return []

    # Convertir HTML en texte
    if progress_cb:
        progress_cb(0.5, f"Conversion de {len(pages)} pages en texte...")

    results = []
    for i, page in enumerate(pages):
        text = _clean_html_to_text(page.get("html_content", ""))

        # Ajouter le titre en en-tête du texte
        if text:
            text = f"# {page['title']}\n\n{text}"

        results.append({
            "id": page["id"],
            "title": page["title"],
            "url": page["url"],
            "path": page["path"],
            "text": text,
        })

        if progress_cb and (i + 1) % 10 == 0:
            progress = 0.5 + (0.4 * (i + 1) / len(pages))
            progress_cb(progress, f"Conversion: {i + 1}/{len(pages)} pages...")

    if progress_cb:
        progress_cb(1.0, f"Extraction terminée: {len(results)} pages")

    return results


def group_pages_by_section(pages: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Groupe les pages par leur section racine (premier ancêtre).

    Args:
        pages: Liste de pages avec le champ 'path' (format: "Section > Sous-section > Page")

    Returns:
        Dict avec clé = nom de section, valeur = liste de pages
    """
    sections = {}

    for page in pages:
        path = page.get("path", page.get("title", "Sans titre"))
        # Extraire la section racine (premier élément du chemin)
        parts = [p.strip() for p in path.split(">")]
        root_section = parts[0] if parts else "Racine"

        if root_section not in sections:
            sections[root_section] = []
        sections[root_section].append(page)

    # Trier les sections par nom
    return dict(sorted(sections.items()))


def get_space_info(
    base_url: str,
    space_key: str,
    username: str,
    password: str,
    verify_ssl: bool = True,
    context_path: str = "",
) -> Optional[Dict[str, Any]]:
    """
    Récupère les informations d'un espace.

    Args:
        context_path: Chemin de contexte pour Confluence Server (ex: /confluence, /wiki)

    Returns:
        Dict avec 'key', 'name', 'description' ou None si non trouvé
    """
    api_url = _build_api_url(base_url, f"space/{space_key}", context_path)

    try:
        response = requests.get(
            api_url,
            auth=_get_auth(username, password),
            timeout=10,
            verify=verify_ssl,
        )

        if response.status_code == 200:
            data = response.json()
            return {
                "key": data.get("key", space_key),
                "name": data.get("name", ""),
                "description": data.get("description", {}).get("plain", {}).get("value", ""),
            }
        elif response.status_code == 404:
            return None
        else:
            logging.error(f"[confluence] Erreur {response.status_code} pour l'espace {space_key}")
            return None

    except Exception as e:
        logging.error(f"[confluence] Erreur récupération espace {space_key}: {e}")
        return None
