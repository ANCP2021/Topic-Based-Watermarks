import os
from typing import Dict, Any
from postmark.models import Watermarker
from postmark.utils import compute_presence

def watermark_and_detect(
    base_text: str,
    llm: str = "llama-3-8b-chat",  # options: "llama-3-8b", "llama-3-8b-chat", "mistral-7b-inst"
    embedder: str = "nomic",
    inserter: str = "llama-3-8b-chat",  # options: "llama-3-8b", "llama-3-8b-chat", "mistral-7b-inst"
    ratio: float = 0.12,
    iterate: str = "v2",            # options: "v2" (iterative insertion) or None (single pass)
    presence_thresh: float = 0.7,    # detection cosine threshold τ
    max_tokens: int = 1500,
) -> Dict[str, Any]:
    """Watermark `base_text` with PostMark and compute detection presence.

    Returns a dict with keys: text1, list1, text2, list2, score2.
    """
    watermarker = Watermarker(
        llm=llm,
        embedder=embedder,
        inserter=inserter,
        ratio=ratio,
        iterate=iterate,
    )

    res = watermarker.insert_watermark(base_text, max_tokens=max_tokens)
    score2 = compute_presence(res["text2"], res["list2"], threshold=presence_thresh)
    return {
        "text1": res["text1"],
        "list1": res["list1"],
        "text2": res["text2"],
        "list2": res["list2"],
        "score2": score2,
    }


if __name__ == "__main__":
    # Example usage: replace `example_text` with your own string.
    example_text = (
        "It also supports Oracle, MS-Access, and B2Vista. The server runs clustered versions of Unix. It supports Virtual SAN. It has one onboard hard drive and two externally attached hard drives. The server has built-in Web browser and FTP server capabilities. It comes with a Linux-based telnet interface. The user interface is based on the Konsole language. The software supports file and directory browsing. It can be configured to run under different operating system versions. It supports file transfers. It supports file copying and editing. It supports backup and recovery. The software also supports network connectivity and file sharing. The user interface and documentation are in the Linux project’s Make it Web (http://www.linux.org/makeitweb/). The eServer 325 supports hardware and software RAID. It supports mirrored drives and RAID 5. It supports RAID 3, HEAF, and RAID 5. It supportsatters and RAID 5’s identical mirrored copies. Cluster servers can support multiple users."
    )
    result = watermark_and_detect(example_text)
    print("Watermarked text (text2):\n")
    print(result["text2"])
    print("\nWatermark words (list2):")
    print(result["list2"])
    print("\nPresence score (score2):", result["score2"])


