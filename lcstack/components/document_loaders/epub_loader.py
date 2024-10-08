import logging
from pathlib import Path
from typing import Iterator, Optional, Union

from langchain_core.documents import Document
from langchain_community.document_loaders.blob_loaders.schema import Blob

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)


class EpubLibEpubLoader(BaseLoader):
    """Load `epub` files and parse them with `ebooklib`."""

    def __init__(
        self,
        file_path: Union[str, Path],
        bodywidth: Optional[int] = None,
        load_images: bool = False,
    ) -> None:
        """initialize with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object.

        Args:
            file_path: The path to the file to load.
            open_encoding: The encoding to use when opening the file.
            bodywidth: The width of the body. 0 means no limit.
        """
        try:
            import ebooklib  # type: ignore
        except ImportError:
            raise ImportError(
                "ebooklib package not found, please install it with "
                "`pip install ebooklib`"
            )

        self.file_path = file_path
        self.bodywidth = bodywidth
        self.load_images = load_images

    def lazy_load(self) -> Iterator[Document]:
        """Load HTML document into document objects."""
        import html2text
        from ebooklib import epub, ITEM_DOCUMENT  # support html only for now

        book: epub.EpubBook = epub.read_epub(self.file_path)

        pages = book.get_items_of_type(ITEM_DOCUMENT)
        book_creator = [creator[0] for creator in book.get_metadata("DC", "creator")]
        index = 0
        # parse toc, Link or Section
        # we parse the top level toc only
        toc = []

        def _get_toc(links):
            for link in links:
                if isinstance(link, tuple) or isinstance(link, list):
                    _section, _links = link
                    title = _section.title
                    parts = _section.href.split("#")
                    uid = parts[1] if len(parts) > 1 else None
                    href = parts[0]
                    toc.append((href, (title, uid)))
                else:
                    toc.append((link.href.split("#")[0], (link.title, link.uid)))

        _get_toc(book.toc)
        toc_map = dict(toc)
        for page in pages:
            # page: epub.EpubHtml = page
            html = page.content.decode("utf-8")
            text = html2text.html2text(html, bodywidth=self.bodywidth)
            # with markdown, append a `#` for all the headings, for example: `#` -> `##`, `###` -> `####`
            new_lines = []
            for line in text.split("\n"):
                # max heading level is 6
                if line.startswith("#") and line[:6] != "######":
                    new_lines.append(line.replace("#", "##", 1))
                else:
                    new_lines.append(line)
            text = "\n".join(new_lines)

            index += 1
            toc_title, toc_uid = toc_map.get(page.file_name, (None, None))
            metadata = {
                "source": str(self.file_path),
                "book_name": book.title,
                "book_uid": book.uid,
                "book_creator": ", ".join(book_creator),
                "title": toc_title or page.title or page.file_name.split(".")[0],
                "uid": toc_uid or page.id,
                "file": page.file_name,
                "type": page.__class__.__name__,
                "media_type": page.media_type,
                "lang": page.lang or book.language,
                "direction": page.direction or "",
                "index": index,
                "is_chapter": page.is_chapter(),
            }
            yield Document(page_content=text, metadata=metadata)

        # if self.load_images:
        #     # also, load images
        #     for blob in self.yield_blobs():
        #         yield blob
        return

    def yield_blobs(self, book=None) -> Iterator[Blob]:
        """Yield images in the epub."""

        from ebooklib import epub, ITEM_IMAGE

        if book is None:
            book: epub.EpubBook = epub.read_epub(self.file_path)
        else:
            book: epub.EpubBook = book
        images = book.get_items_of_type(ITEM_IMAGE)
        book_creator = [creator[0] for creator in book.get_metadata("DC", "creator")]
        for image in images:
            image: epub.EpubImage = image
            metadata = {
                "source": str(self.file_path),
                "file": image.file_name,
                "type": image.__class__.__name__,
                "media_type": image.media_type,
                "book_name": book.title,
                "book_uid": book.uid,
                "book_creator": ", ".join(book_creator),
            }
            blob = Blob.from_data(
                image.content,
                mime_type=image.media_type,
                encoding="binary",
                path=image.file_name,
                metadata=metadata,
            )
            yield blob


if __name__ == "__main__":
    epub_file = "../data/An Introduction to Critical Thinking.epub"
    epub_file = "../data/Apache Iceberg The Definitive Guide.epub"
    epub_file = "../data/Data Management at Scale 2nd Edition.epub"
    root_path = "../data/books"
    import os
    import sys

    if len(sys.argv) > 1:
        epub_file = sys.argv[1]
    if len(sys.argv) > 2:
        root_path = sys.argv[2]
    loader = EpubLibEpubLoader(epub_file, bodywidth=0)
    book_name = os.path.split(epub_file)[-1]
    book_name, _ = os.path.splitext(book_name)
    book_path = os.path.join(root_path, book_name)
    if not os.path.exists(book_path):
        os.makedirs(book_path)
    with open(f"{book_path}/{book_name}.md", "w", encoding="utf-8") as f:
        for doc in loader.load():
            f.write(doc.page_content)
            f.write("\n\n")
    # move the images out of the epub (./Images/)
    image_path = os.path.join(book_path, "Images")
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    for blob in loader.yield_blobs():
        with open(os.path.join(book_path, blob.metadata["file"]), "wb") as f:
            f.write(blob.data)
