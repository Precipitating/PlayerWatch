
## 🧠 Methods Available
[Ball detection model](https://drive.google.com/uc?id=1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V)

[Player detection model](https://drive.google.com/uc?id=17PXFNlx-jI7VjVo_vQnB1sONjRyvoB-q)
### 1. YOLO and YOLO
**Tracking**:  
Players and ball are tracked using trained YOLO models.

**Possession Logic**:
- For each frame, a list of who is possessing the ball is buffered.
- If a player maintains possession for _x_ frames, the **start timestamp** is set.
- When another player possesses the ball for _x_ frames, the **end timestamp** is recorded.
- This process continues throughout the video.

---

### 2. YOLO + SAM 2
**Tracking**:
- **Players**: YOLO  
- **Ball**: SAM 2 (with assistance from Florence 2 to detect the ball in the first frame)

**Possession Logic**:  
Same as the YOLO-only method but with higher accuracy.

**Trade-offs**:
- Significantly slower processing
- Very high GPU usage
- More of a technical showcase than practical for production use

---

### 3. Audio Method
**Overview**:  
The fastest method, ideal when match commentary is available.

**Detection Logic**:
- Uses `faster-whisper` to transcribe audio.
- Detects when a player’s name is mentioned to mark the **start** of possession.
- The **end** is set to a fixed time after the mention.

**Performance**:
- Processes a ~50-minute video in just a few minutes.
- Highly effective and efficient when commentary is present.
