# SPDX-License-Identifier: Apache-2.0
"""Pinned GGUF model and projector artifacts for built-in Persona models."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PersonaGGUFFile:
    """Describe one GGUF artifact with its exact integrity metadata."""

    filename: str
    size: int
    sha256: str


@dataclass(frozen=True, slots=True)
class PersonaGGUFVariant:
    """Describe one exact model and vision-projector package variant."""

    repo_id: str
    revision: str
    request_quantization: str
    quantization: str
    model: PersonaGGUFFile
    projector: PersonaGGUFFile


@dataclass(frozen=True, slots=True)
class _PersonaGGUFPackage:
    """Describe one repository's verified quant files and shared projector."""

    repo_id: str
    revision: str
    model_stem: str
    q4_size: int
    q4_sha256: str
    q8_size: int
    q8_sha256: str
    projector_size: int
    projector_sha256: str

    def variants(self) -> dict[str, PersonaGGUFVariant]:
        """Build the supported four-bit and eight-bit runtime variants."""
        projector = PersonaGGUFFile(
            filename=f"{self.model_stem}.mmproj-Q8_0.gguf",
            size=self.projector_size,
            sha256=self.projector_sha256,
        )
        return {
            "4bit": PersonaGGUFVariant(
                repo_id=self.repo_id,
                revision=self.revision,
                request_quantization="4bit",
                quantization="Q4_K_M",
                model=PersonaGGUFFile(
                    filename=f"{self.model_stem}.Q4_K_M.gguf",
                    size=self.q4_size,
                    sha256=self.q4_sha256,
                ),
                projector=projector,
            ),
            "8bit": PersonaGGUFVariant(
                repo_id=self.repo_id,
                revision=self.revision,
                request_quantization="8bit",
                quantization="Q8_0",
                model=PersonaGGUFFile(
                    filename=f"{self.model_stem}.Q8_0.gguf",
                    size=self.q8_size,
                    sha256=self.q8_sha256,
                ),
                projector=projector,
            ),
        }


_PACKAGES = (
    (
        "huihui-ai/Huihui-Qwen3-VL-4B-Instruct-abliterated",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Huihui-Qwen3-VL-4B-Instruct-abliterated-GGUF",
            revision="abf99adf548eb2aa6f9a3d56f976968996959e7f",
            model_stem="Huihui-Qwen3-VL-4B-Instruct-abliterated",
            q4_size=2_497_282_496,
            q4_sha256="20d58872549c13f5476622d2385027e1b346298558822eca8aa33cb8846d5765",
            q8_size=4_280_406_976,
            q8_sha256="aba89057758004b0349b3b8ccc7f453386f9389ed6dea71a716be7ab5ede9a92",
            projector_size=453_974_624,
            projector_sha256="cd4503cef5b48c55da5f04edd06e88f3c10344129f19bd9ebffdab32d0590a02",
        ),
    ),
    (
        "Qwen/Qwen3-VL-8B-Instruct",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Qwen3-VL-8B-Instruct-GGUF",
            revision="499df3a68793b65eaec9035247e32f5cffa346b1",
            model_stem="Qwen3-VL-8B-Instruct",
            q4_size=5_027_785_344,
            q4_sha256="659ea2c20fe881bba78e6c7a41dc206ab4de993b3df80f554c18b098b673792e",
            q8_size=8_709_520_000,
            q8_sha256="2370e7ec38c2ff0ee8b5aefaccbb86018a7bf44381a4970a0039832a9016ca2d",
            projector_size=752_289_824,
            projector_sha256="7e848608a42de7e514803662b922f3835ce4191cde2709af1a2882d682c67718",
        ),
    ),
    (
        "huihui-ai/Huihui-Qwen3-VL-8B-Instruct-abliterated",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Huihui-Qwen3-VL-8B-Instruct-abliterated-GGUF",
            revision="03f1e8f28be1177d430866b31e0bc9a939c669d6",
            model_stem="Huihui-Qwen3-VL-8B-Instruct-abliterated",
            q4_size=5_027_785_728,
            q4_sha256="c9592053aab8874021ecd3f80e45cc12dd167454bebfb215a36bdf690f22dc52",
            q8_size=8_709_520_384,
            q8_sha256="2a80cab9a07998164351e47c67380cbd10df95afbe8f3726688b82723689c64c",
            projector_size=752_290_176,
            projector_sha256="8083b2ccf0351ce673fca1dd5667ad1fd49861c72dccebb16f3dd32aa6c42fb6",
        ),
    ),
    (
        "Qwen/Qwen3-VL-8B-Thinking",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Qwen3-VL-8B-Thinking-GGUF",
            revision="e92647a6203cbcb918c7efe7b08b245db7291681",
            model_stem="Qwen3-VL-8B-Thinking",
            q4_size=5_027_785_248,
            q4_sha256="e495720e0e823fa8fb9f082a4012ccb412deebc9f03fa83dac7e6e15b477c693",
            q8_size=8_709_519_904,
            q8_sha256="00673fa25542eda8b95d4c01f8f73cf864fe64640fd83bfddd78573eee2d2537",
            projector_size=752_289_824,
            projector_sha256="09add99dd97f82aa04bd39ec2460e099b9bfb29b7f20e7605346fc85b6dfa87e",
        ),
    ),
    (
        "huihui-ai/Huihui-Qwen3-VL-8B-Thinking-abliterated",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Huihui-Qwen3-VL-8B-Thinking-abliterated-GGUF",
            revision="7b20225df54c856f085e3d0d120ca6cab3aa10b6",
            model_stem="Huihui-Qwen3-VL-8B-Thinking-abliterated",
            q4_size=5_027_785_632,
            q4_sha256="478f1c18f5cd3bcbd1a1e6ec36e30451d2ad39566161176f019c678c1a30090b",
            q8_size=8_709_520_288,
            q8_sha256="2af6547ea5630a4e9dd7267815ce7a0358ef2ed426c0d4fd02c4493814bb37a4",
            projector_size=752_290_176,
            projector_sha256="1cadefa6bfa43cf64b34d4e87a3a1bb6ca3f9ed6261b04010a5e7ecbfc6f0023",
        ),
    ),
    (
        "Qwen/Qwen3.5-4B",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Qwen3.5-4B-GGUF",
            revision="85aed52869ec5ef32259f80686311048bc260494",
            model_stem="Qwen3.5-4B",
            q4_size=2_708_804_800,
            q4_sha256="51eafbc127f35598c8f1d2ec58b2520d6126c7d1195c4eca26832e63a2939d39",
            q8_size=4_482_403_520,
            q8_sha256="722af6f11b1f8052ae4854a82947e8f37102275f6b0edc402146144aefa840ab",
            projector_size=366_894_656,
            projector_sha256="40a4f07d7bbdbb43011d6cf35ef751e4b1829ff47ee8aa4964c6296f571725ad",
        ),
    ),
    (
        "huihui-ai/Huihui-Qwen3.5-4B-abliterated",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Huihui-Qwen3.5-4B-abliterated-GGUF",
            revision="9d3aebb998fcf084086a3d1c60c31035fcfbf8e1",
            model_stem="Huihui-Qwen3.5-4B-abliterated",
            q4_size=2_707_514_688,
            q4_sha256="423f10b6ec2d99c3378143d7cd3b80eb4887b3ed92103103ac59173b404f4f7c",
            q8_size=4_482_403_648,
            q8_sha256="1e78f9b48c8b69242fa45bbbbeb439ed818a71381a6362130d62a38561c3756d",
            projector_size=366_894_720,
            projector_sha256="650e2af8754b8b10e21debdbaf352cab43cd2435ec8ed56b893b9b581ae4a95c",
        ),
    ),
    (
        "Qwen/Qwen3.5-9B",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Qwen3.5-9B-GGUF",
            revision="664ad08c870c0a6e80798eb4a98e2335ec6e93b4",
            model_stem="Qwen3.5-9B",
            q4_size=5_627_045_120,
            q4_sha256="9fa52e37c8293ee02cf277634f8d3a392dc3239b8ccef2315b4c8c9ef223e7fb",
            q8_size=9_527_502_080,
            q8_sha256="e7057b92e524bb7bdf824f84b268b4ed64ea0336b51cf442afec22b2a996e132",
            projector_size=624_229_696,
            projector_sha256="6ae4061987abccb86185f3b4e2ae0e4c6375bfd3dc98c4e473eda7545877155c",
        ),
    ),
    (
        "huihui-ai/Huihui-Qwen3.5-9B-abliterated",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Huihui-Qwen3.5-9B-abliterated-GGUF",
            revision="419ca34f424adbd9276c09e0271cb6b718e8da1d",
            model_stem="Huihui-Qwen3.5-9B-abliterated",
            q4_size=5_627_045_248,
            q4_sha256="ea1858ef4dc4b648b8dbb44612962a0333e945060dd0545ac0f28d7c4416e4b3",
            q8_size=9_527_502_208,
            q8_sha256="9df7ce00f4df74344dd788ade97c412ef9850e58b44ff337e56fbf07be24a97f",
            projector_size=624_229_760,
            projector_sha256="cf54897b6bbfa89c809fdc2b8236e0977bed70760b39ee8f11d87296e1d0e046",
        ),
    ),
    (
        "google/gemma-4-E2B-it",
        _PersonaGGUFPackage(
            repo_id="mradermacher/gemma-4-E2B-it-GGUF",
            revision="cb7ceaccb7b0973a17ab6d39fe6d2b11987ea8b5",
            model_stem="gemma-4-E2B-it",
            q4_size=3_427_878_336,
            q4_sha256="7e6e3d87757ca0560a4b0be80dc0bb480bb793b749b317f7aeda81b5ea8260a0",
            q8_size=4_967_495_104,
            q8_sha256="803d2049854ebff08a199f9dfce0c1f25688127c63da5a0c4a8376ccba822c91",
            projector_size=557_367_808,
            projector_sha256="4e1b21a6ddf52e07387be624523c899591d463cd3051819eb6b8acad3f2cc6a6",
        ),
    ),
    (
        "huihui-ai/Huihui-gemma-4-E2B-it-abliterated",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Huihui-gemma-4-E2B-it-abliterated-GGUF",
            revision="0538cae708ae060ba4fff720297b79c672c339ff",
            model_stem="Huihui-gemma-4-E2B-it-abliterated",
            q4_size=3_427_874_272,
            q4_sha256="55044454df35c2e3b9f4ca7a52b27cc7b057b3b3547cd114183196c31f5062a8",
            q8_size=4_967_491_040,
            q8_sha256="c127fb9c6f3867969ceb79078e30eae3c62603aea92598400ac5a93bb2936431",
            projector_size=557_368_128,
            projector_sha256="030b1a496787c3ca9240552a3d5488db0ba50c6d6cdc878793513340d736765c",
        ),
    ),
    (
        "google/gemma-4-E4B-it",
        _PersonaGGUFPackage(
            repo_id="mradermacher/gemma-4-E4B-it-GGUF",
            revision="bccd4df108bbde6cd495f2594c17a5182bc59304",
            model_stem="gemma-4-E4B-it",
            q4_size=5_335_290_304,
            q4_sha256="e5555564868766605415f0e1401feac79c9327333c051521934c90aea67df897",
            q8_size=8_031_240_640,
            q8_sha256="0802b70ad2e46c1db03b45504cd3feedb5d329b24ad98078c5418d14674c4aaa",
            projector_size=559_874_560,
            projector_sha256="2d5aa169b4b61a9281e9106acbe73d8f2437e316ba9657f65eb3e3332f5c598a",
        ),
    ),
    (
        "huihui-ai/Huihui-gemma-4-E4B-it-abliterated",
        _PersonaGGUFPackage(
            repo_id="mradermacher/Huihui-gemma-4-E4B-it-abliterated-GGUF",
            revision="580a4a13e9633a0f123befcfab222cfcd9a36b77",
            model_stem="Huihui-gemma-4-E4B-it-abliterated",
            q4_size=5_335_286_272,
            q4_sha256="d793116ec11cacac0228e7c907a40e9765969d76dd45708f0f638df008718f87",
            q8_size=8_031_236_608,
            q8_sha256="8bb670d57c5dc042c6ef9571f8b86674b4d1835e55708ae5ac5b40aa16687036",
            projector_size=559_874_880,
            projector_sha256="46f248feeb9288819cdd6a239dfa42335780dcf085165247cbe2c56cae403839",
        ),
    ),
)

PERSONA_GGUF_VARIANTS_BY_MODEL = {
    model_id: package.variants() for model_id, package in _PACKAGES
}

RAON_GGUF_REPO_ID = "KRAFTON/Raon-1-nano-gguf"
RAON_GGUF_REVISION = "435596f1220a24df3a6ad19b36998b606df3385d"
RAON_GGUF_FILES = {
    "F16": PersonaGGUFFile(
        filename="raon-1-nano-F16.gguf",
        size=6_274_887_040,
        sha256="37d9b28f2516c52743685e9a39708ba4bde73c478374a31d54e55d88943bb1bb",
    ),
    "F16_MTP": PersonaGGUFFile(
        filename="raon-1-nano-F16_MTP.gguf",
        size=6_384_592_256,
        sha256="a00a0ab3d177962d957dfe5e190d189ce5d606df324a969d04c8add37cabb8b4",
    ),
    "Q4_K_M": PersonaGGUFFile(
        filename="raon-1-nano-Q4_K_M.gguf",
        size=2_082_677_632,
        sha256="90e9ea83613626457c1249bb85e8d8471dd8874e62b4fa707d40c96737d764b6",
    ),
    "Q4_K_M_MTP": PersonaGGUFFile(
        filename="raon-1-nano-Q4_K_M_MTP.gguf",
        size=2_145_934_208,
        sha256="f83a4f119a5099b72a2cd354de9eeae8f90f6e301680b38628b8fd01221d7b5c",
    ),
    "Q6_K": PersonaGGUFFile(
        filename="raon-1-nano-Q6_K.gguf",
        size=2_646_016_384,
        sha256="98bb590470a9e504b8cd5715683a07c8c9680810d629dde485bc64ff21a6e634",
    ),
    "Q6_K_MTP": PersonaGGUFFile(
        filename="raon-1-nano-Q6_K_MTP.gguf",
        size=2_709_272_960,
        sha256="38af261a61e1f4782348bceb69dfabb662a6294ac0e79b822a9b56ebcbd74186",
    ),
    "Q8_0": PersonaGGUFFile(
        filename="raon-1-nano-Q8_0.gguf",
        size=3_391_016_320,
        sha256="5b4d73eaace1a96f872af4b7e7650bdc8a13389f7ac3da9eeebe06e146d22421",
    ),
    "Q8_0_MTP": PersonaGGUFFile(
        filename="raon-1-nano-Q8_0_MTP.gguf",
        size=3_454_272_896,
        sha256="b324a8aa8d089f10d5672e8c8bef1c4a30caa93aa55571540037f5771b1a90b7",
    ),
}
