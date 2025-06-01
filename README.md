

# FMTrack: Frequency-aware Interaction and Multi-Expert Fusion for RGB-T Tracking

Abstract — Recently, RGB-T tracking has received increasing attention due to its robustness. 
However, existing RGB-T trackers mainly use cross-attention for modal feature interaction, 
limiting the utilization of complementary information. In addition, these trackers employ fixed dominant-auxiliary 
paradigms for feature fusion, ignoring modal quality fluctuations. To address these issues, we propose FMTrack, 
an effective framework for fully capturing complementary information. FMTrack consists of two key components, 
a frequency-aware interaction network (FIN) and a multi-expert fusion module (MEFM). To emphasize the valuable 
information in each modality, FIN utilizes frequency masks to perform high-pass and low-pass filtering on RGB and 
TIR data. FIN explicitly establishes cross-modal interactions via frequency domain learning, which facilitates the 
sharing of complementary information. Besides, MEFM extracts diverse features via the differentiated expert network 
and then adjusts feature combinations according to modal reliability, achieving deep understanding and flexible fusion 
of multimodal data. With FIN and MEFM, FMTrack makes full use of the advantageous information of each modality to
highlight target representations, thus improving performance in complex scenes. Extensive experiments on four popular 
RGBT tracking datasets (LasHeR, VTUAV, RGBT234, and RGBT210) show that our FMTrack achieves leading performance.
The code is available at https://github.com/xyl-507/FMTrack.

---
Note:
Our paper is under peer review. Complete code and models will be released once accepted.

