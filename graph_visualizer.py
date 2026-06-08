import json
from typing import List, Optional

from pyvis.network import Network
from scipy.sparse import csr_matrix


class GraphVisualizer:
    # колір вузла = книга Євангелія
    BOOK_COLORS = {
        "Matthew": "#e74c3c",
        "Mark": "#3498db",
        "Luke": "#2ecc71",
        "John": "#f39c12",
    }
    DEFAULT_NODE_COLOR = "#95a5a6"
    QUERY_NODE_COLOR = "#ffd700"

    def __init__(
        self,
        canvas_height: str = "100vh",
        canvas_width: str = "100%",
        background_color: str = "#0e0e15",
        font_color: str = "#f5f5f5",
        show_inter_verse_edges: bool = False,
    ):
        self._canvas_height = canvas_height
        self._canvas_width = canvas_width
        self._background_color = background_color
        self._font_color = font_color
        self._show_inter_verse_edges = show_inter_verse_edges
        self._html_template = ""
        self._json_payload = ""
        self.success = False

    def generate_layout(
        self,
        nodes: List[dict],
        matrix: Optional[csr_matrix],
        output_filename: str = "output.html",
        query_text: str = "",
    ) -> bool:
        try:
            network = Network(
                height=self._canvas_height,
                width=self._canvas_width,
                bgcolor=self._background_color,
                font_color=self._font_color,
                directed=False,
                notebook=False,
                cdn_resources="in_line",
            )

            query_label = query_text if len(query_text) <= 60 else query_text[:57] + "..."
            network.add_node(
                "query",
                label=query_label,
                title=f"Пошуковий запит\n{query_text}",
                color={"background": self.QUERY_NODE_COLOR, "border": "#ffffff"},
                size=45,
                shape="star",
                physics=False,
                x=0,
                y=0,
                fixed=True,
                font={"size": 20, "bold": True, "color": "#ffffff"},
                borderWidth=3,
            )

            edges_payload = []

            for idx, node_data in enumerate(nodes):
                book = node_data.get("book", "Unknown")
                chapter = node_data.get("chapter", 0)
                verse = node_data.get("verse", 0)
                text = node_data.get("text", "")
                similarity = float(node_data.get("similarity", 0.0))

                color = self.BOOK_COLORS.get(book, self.DEFAULT_NODE_COLOR)
                verse_id = f"v{idx}"
                node_size = 8.0 + similarity * 32.0

                label = f"{book} {chapter}:{verse}"
                tooltip = f"{book} {chapter}:{verse}\nСхожість {similarity:.4f}\n\n{text}"

                network.add_node(
                    verse_id,
                    label=label,
                    title=tooltip,
                    color=color,
                    size=node_size,
                    font={"size": 11, "color": self._font_color},
                )

                clamped_sim = max(0.0, min(1.0, similarity))
                edge_length = 80.0 + (1.0 - clamped_sim) * 650.0
                edge_width = 0.6 + clamped_sim * 9.0
                edge_opacity = min(1.0, 0.25 + clamped_sim * 0.85)

                network.add_edge(
                    "query",
                    verse_id,
                    length=edge_length,
                    width=edge_width,
                    color={"color": color, "opacity": edge_opacity},
                    smooth=False,
                    physics=True,
                )

                edges_payload.append({
                    "source": "query",
                    "target": verse_id,
                    "similarity": similarity,
                    "length": edge_length,
                    "width": edge_width,
                })

            if self._show_inter_verse_edges and matrix is not None:
                coo = matrix.tocoo()
                for row, col, weight in zip(coo.row, coo.col, coo.data):
                    if row >= col:
                        continue
                    weight_value = float(weight)
                    if weight_value <= 0:
                        continue
                    network.add_edge(
                        f"v{int(row)}",
                        f"v{int(col)}",
                        width=0.3 + weight_value * 2.0,
                        color={"color": "#555566", "opacity": 0.25},
                        smooth=False,
                        physics=False,
                    )

            network.set_options(self._physics_options())
            self._json_payload = json.dumps(
                {"query": query_text, "nodes": nodes, "edges": edges_payload},
                ensure_ascii=False,
                indent=2,
            )
            network.write_html(output_filename, notebook=False, open_browser=False)
            self._html_template = output_filename
            self.success = True
            return True

        except TypeError as exc:
            self.success = False
            raise TypeError(f"Помилка серіалізації при побудові графа: {exc}") from exc
        except OSError as exc:
            self.success = False
            raise OSError(f"Помилка запису файлу {output_filename}: {exc}") from exc

    def _physics_options(self) -> str:
        options = {
            "physics": {
                "enabled": True,
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                    "gravitationalConstant": -80,
                    "centralGravity": 0.005,
                    "springLength": 200,
                    "springConstant": 0.08,
                    "damping": 0.55,
                    "avoidOverlap": 0.6,
                },
                "minVelocity": 0.5,
                "maxVelocity": 30,
                "timestep": 0.4,
                "stabilization": {"enabled": True, "iterations": 1200, "fit": True},
            },
            "nodes": {
                "borderWidth": 1,
                "shape": "dot",
                "shadow": {"enabled": True, "color": "rgba(0,0,0,0.5)", "size": 8},
                "font": {"strokeWidth": 2, "strokeColor": "#0e0e15"},
            },
            "edges": {"smooth": False, "selectionWidth": 2},
            "interaction": {
                "hover": True,
                "tooltipDelay": 80,
                "navigationButtons": True,
                "keyboard": True,
                "zoomView": True,
                "dragView": True,
            },
        }
        return json.dumps(options)

    def export_json_payload(self, json_path: str) -> None:
        if not self._json_payload:
            return
        with open(json_path, "w", encoding="utf-8") as fh:
            fh.write(self._json_payload)
