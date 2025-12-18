

<script setup>
import { ref, computed } from 'vue'
import axios from 'axios'

const API_URL = 'http://localhost:8000'

// --- LÓGICA DE TIEMPO ---
const selectedFile = ref(null)
const isAnalyzing = ref(false)
const timelineData = ref([])
const totalErrors = ref(0)
const dominantActivityId = ref(0) // Agregado para cálculo actividad dominante

const comparisonLeft = ref(null)
const comparisonRight = ref(null)

const ACTIVITY_COLORS = {
  1: '#95a5a6', // Standing
  2: '#7f8c8d', // Sitting
  3: '#bdc3c7', // Lying
  4: '#2ecc71', // Walking
  5: '#27ae60', // Climbing
  6: '#f1c40f', // Waist bends
  7: '#e67e22', // Arms up
  8: '#d35400', // Crouching
  9: '#3498db', // Cycling
  10: '#2980b9', // Jogging
  11: '#9b59b6', // Running
  12: '#8e44ad', // Jumps
  0: '#ecf0f1'   // Null
}

const ACTIVITY_NAMES = {
  1: 'De pie',
  2: 'Sentado y relajado',
  3: 'Acostado',
  4: 'Caminando',
  5: 'Subiendo escaleras',
  6: 'Flexión de cintura',
  7: 'Elevación de brazos',
  8: 'Flexión de rodillas',
  9: 'Ciclismo',
  10: 'Trotar',
  11: 'Correr',
  12: 'Saltos',
  0: 'Nulo/Desconocido'
}

const handleFileUpload = (event) => {
  selectedFile.value = event.target.files[0]
}

const selectSegment = (seg) => {
    if (!comparisonLeft.value) {
        comparisonLeft.value = seg
    } else if (!comparisonRight.value) {
        // Evitar seleccionar el mismo
        if (comparisonLeft.value.segment_id !== seg.segment_id) {
            comparisonRight.value = seg
        }
    } else {
        // Si ya hay dos, reemplazamos el derecho para comparar rápido
        comparisonRight.value = seg
    }
}

const clearSelection = () => {
    comparisonLeft.value = null
    comparisonRight.value = null
}

const getFeatureValue = (features, channelIdx, statIdx) => {
    // Index = (Channel * 7) + Stat
    const idx = (channelIdx * 7) + statIdx
    if (features && features.length > idx) {
        return features[idx].toFixed(2)
    }
    return '-'
}

// Todos los 26 canales para inspección (Español)
const ALL_CHANNELS = [
    // Pecho (0-2)
    { name: 'Acel. Pecho X', idx: 0, group: 'Pecho' },
    { name: 'Acel. Pecho Y', idx: 1, group: 'Pecho' },
    { name: 'Acel. Pecho Z', idx: 2, group: 'Pecho' },
    // Tobillo Acel (3-5)
    { name: 'Acel. Tobillo X', idx: 3, group: 'Tobillo' },
    { name: 'Acel. Tobillo Y', idx: 4, group: 'Tobillo' },
    { name: 'Acel. Tobillo Z', idx: 5, group: 'Tobillo' },
    // Tobillo Giro (6-8)
    { name: 'Giro. Tobillo X', idx: 6, group: 'Tobillo' },
    { name: 'Giro. Tobillo Y', idx: 7, group: 'Tobillo' },
    { name: 'Giro. Tobillo Z', idx: 8, group: 'Tobillo' },
    // Tobillo Mag (9-11)
    { name: 'Mag. Tobillo X', idx: 9, group: 'Tobillo' },
    { name: 'Mag. Tobillo Y', idx: 10, group: 'Tobillo' },
    { name: 'Mag. Tobillo Z', idx: 11, group: 'Tobillo' },
    // Brazo Acel (12-14)
    { name: 'Acel. Brazo X', idx: 12, group: 'Brazo' },
    { name: 'Acel. Brazo Y', idx: 13, group: 'Brazo' },
    { name: 'Acel. Brazo Z', idx: 14, group: 'Brazo' },
    // Brazo Giro (15-17)
    { name: 'Giro. Brazo X', idx: 15, group: 'Brazo' },
    { name: 'Giro. Brazo Y', idx: 16, group: 'Brazo' },
    { name: 'Giro. Brazo Z', idx: 17, group: 'Brazo' },
    // Brazo Mag (18-20)
    { name: 'Mag. Brazo X', idx: 18, group: 'Brazo' },
    { name: 'Mag. Brazo Y', idx: 19, group: 'Brazo' },
    { name: 'Mag. Brazo Z', idx: 20, group: 'Brazo' },
    // Magnitudes (21-25)
    { name: 'Mag. Acel. Tobillo', idx: 21, group: 'Calculado' },
    { name: 'Mag. Giro. Tobillo', idx: 22, group: 'Calculado' },
    { name: 'Mag. Acel. Brazo', idx: 23, group: 'Calculado' },
    { name: 'Mag. Giro. Brazo', idx: 24, group: 'Calculado' },
    { name: 'Mag. Acel. Pecho', idx: 25, group: 'Calculado' }
]

// Sensores "Relevant" para vista simplificada
const RELEVANT_CHANNELS = ALL_CHANNELS.filter(c => 
    [0, 1, 3, 14, 16, 23, 25].includes(c.idx)
)

const showAllSensors = ref(false)

const currentInspectionChannels = computed(() => {
    return showAllSensors.value ? ALL_CHANNELS : RELEVANT_CHANNELS
})

const analyzeLog = async () => {
  if (!selectedFile.value) return
  
  isAnalyzing.value = true
  timelineData.value = []
  clearSelection()
  totalErrors.value = 0
  
  const formData = new FormData()
  formData.append('file', selectedFile.value)
  
  try {
    const response = await axios.post(`${API_URL}/predict/log`, formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    timelineData.value = response.data.timeline
    
    // Auto-seleccionar primer segmento
    if (timelineData.value.length > 0) {
        comparisonLeft.value = timelineData.value[0]
    }
    
    // Calcular Dominante y Errores
    const counts = {}
    let errors = 0
    timelineData.value.forEach(seg => {
        const pid = seg.predicted_id
        counts[pid] = (counts[pid] || 0) + 1
        
        // Contar error si no coinciden y la etiqueta real no es 0 (null)
        if (seg.true_id !== 0 && seg.predicted_id !== seg.true_id) {
            errors++
        }
    })
    totalErrors.value = errors
    
    let maxId = 0
    let maxCount = 0
    for(const [id, count] of Object.entries(counts)) {
        if(count > maxCount && id != 0) {
            maxCount = count
            maxId = id
        }
    }
    dominantActivityId.value = maxId
    
  } catch (error) {
    console.error("Error analizando log:", error)
    alert("Error analizando log. Revise la consola.")
  } finally {
    isAnalyzing.value = false
  }
}

// Highlight Difference Logic
const getDiffClass = (val1, val2) => {
    if (val1 === '-' || val2 === '-') return ''
    const diff = Math.abs(parseFloat(val1) - parseFloat(val2))
    if (diff > 2.0) return 'diff-high' // Significant deviation in Z-score
    if (diff > 1.0) return 'diff-med'
    return ''
}

const getActivityColor = (id) => {
  return ACTIVITY_COLORS[id] || '#000000'
}
</script>

<template>
  <div class="container">
    <header>
        <h1>Detector de Movimientos Humanos</h1>
      <p class="subtitle">Inteligencia Artificial para Reconocimiento de Actividades Humanas</p>
    </header>
    
    <main>
      <!-- Timeline Section -->
      <div class="timeline-section">
         <div class="card">
           <h2>1. Análisis de Archivo Log</h2>
           <p class="instruction-text">Sube un archivo de log (ej. mHealth_subject2.log) para ver qué detecta la IA.</p>
           
           <div class="upload-controls">
             <input type="file" @change="handleFileUpload" accept=".log,.txt,.csv" />
             <button @click="analyzeLog" :disabled="!selectedFile || isAnalyzing" class="analyze-btn">
               {{ isAnalyzing ? 'Analizando...' : 'Subir y Analizar' }}
             </button>
           </div>
           
           <div v-if="timelineData.length > 0">
              <!-- Summary Card -->
               <div class="summary-box">
                  <div class="summary-item">
                    <span class="summary-label">Duración Total</span>
                    <span class="summary-value">{{ (timelineData.length).toFixed(0) }} seg</span>
                  </div>
                  <div class="summary-item">
                    <span class="summary-label">Actividad Dominante</span>
                    <span class="summary-value" :style="{color: getActivityColor(dominantActivityId)}">
                       {{ ACTIVITY_NAMES[dominantActivityId] }}
                    </span>
                  </div>
                   <div class="summary-item">
                    <span class="summary-label">Confusiones detectadas</span>
                    <span class="summary-value error-text">{{ totalErrors }}</span>
                  </div>
               </div>

              <div class="timeline-vis">
                <h3>Línea de Tiempo de Actividad</h3>
                <p class="chart-caption">Predicción  vs. Real . <strong>¡Haz clic para comparar!</strong></p>
                <p class="error-legend"><span class="error-box-sample"></span> Las zonas oscuras indican errores de predicción.</p>
                
                <div class="timeline-row">
                  <span class="row-label">Predicción IA:</span>
                  <div class="timeline-track">
                    <div v-for="seg in timelineData" 
                        :key="'p-'+seg.segment_id"
                        class="timeline-segment"
                        :class="{ 
                            'selected': (comparisonLeft && comparisonLeft.segment_id === seg.segment_id) || (comparisonRight && comparisonRight.segment_id === seg.segment_id),
                            'error-segment': seg.predicted_id !== seg.true_id && seg.true_id !== 0
                        }"
                        :style="{ backgroundColor: getActivityColor(seg.predicted_id), flex: 1 }"
                        :title="`Predicho: ${seg.predicted_label}`"
                        @click="selectSegment(seg)">
                    </div>
                  </div>
                </div>
                
                <div class="timeline-row">
                  <span class="row-label">Etiqueta Real:</span>
                  <div class="timeline-track">
                    <div v-for="seg in timelineData" 
                        :key="'t-'+seg.segment_id"
                        class="timeline-segment"
                        :class="{ 
                             'selected': (comparisonLeft && comparisonLeft.segment_id === seg.segment_id) || (comparisonRight && comparisonRight.segment_id === seg.segment_id),
                             'error-segment': seg.predicted_id !== seg.true_id && seg.true_id !== 0
                        }"
                        :style="{ backgroundColor: getActivityColor(seg.true_id), flex: 1 }"
                        :title="`Real: ${seg.true_label}`"
                        @click="selectSegment(seg)">
                    </div>
                  </div>
                </div>
                
                <div class="legend">
                  <div v-for="(color, id) in ACTIVITY_COLORS" :key="id" class="legend-item" v-show="id!=0">
                      <span class="color-box" :style="{backgroundColor: color}"></span>
                      <span class="label-text">{{ ACTIVITY_NAMES[id] }}</span>
                  </div>
                </div>
              </div>

              <!-- INSPECTION PANEL -->
              <div v-if="comparisonLeft" class="inspection-panel">
                  <div class="panel-header">
                      <h3>🔍 Comparador de Datos (Valores Normalizados)</h3>
                      <div class="header-controls">
                        <label class="toggle-sensors">
                            <input type="checkbox" v-model="showAllSensors"> Mostrar todos los sensores (26)
                        </label>
                        <button class="close-btn" @click="clearSelection">✖ Limpiar</button>
                      </div>
                  </div>
                  
                  <div class="comparison-grid">
                      <!-- LEFT SIDE -->
                      <div class="data-column">
                           <div class="column-header" :style="{ borderBottom: '4px solid ' + getActivityColor(comparisonLeft.predicted_id)}">
                                <span>Segundo #{{ comparisonLeft.start_time_idx / 50 }}</span>
                                <strong :style="{color: getActivityColor(comparisonLeft.predicted_id)}">{{ comparisonLeft.predicted_label }}</strong>
                                <small v-if="comparisonLeft.predicted_id !== comparisonLeft.true_id" class="error-badge">
                                    (Real: {{ comparisonLeft.true_label }})
                                </small>
                           </div>
                           
                           <table class="features-table">
                              <thead>
                                  <tr>
                                      <th>Sensor</th>
                                      <th>Media</th>
                                      <th>Std (Intensidad)</th>
                                      <th>Energía</th>
                                  </tr>
                              </thead>
                              <tbody>
                                  <tr v-for="channel in currentInspectionChannels" :key="channel.name" 
                                      :class="comparisonRight ? getDiffClass(getFeatureValue(comparisonLeft.features, channel.idx, 0), getFeatureValue(comparisonRight.features, channel.idx, 0)) : ''">
                                      <td>{{ channel.name }}</td>
                                      <td>{{ getFeatureValue(comparisonLeft.features, channel.idx, 0) }}</td>
                                      <td>{{ getFeatureValue(comparisonLeft.features, channel.idx, 1) }}</td>
                                      <td>{{ getFeatureValue(comparisonLeft.features, channel.idx, 6) }}</td>
                                  </tr>
                              </tbody>
                           </table>
                      </div>

                      <!-- RIGHT SIDE (If Selected) -->
                      <div v-if="comparisonRight" class="data-column">
                           <div class="column-header" :style="{ borderBottom: '4px solid ' + getActivityColor(comparisonRight.predicted_id)}">
                                <span>Segundo #{{ comparisonRight.start_time_idx / 50 }}</span>
                                <strong :style="{color: getActivityColor(comparisonRight.predicted_id)}">{{ comparisonRight.predicted_label }}</strong>
                                <small v-if="comparisonRight.predicted_id !== comparisonRight.true_id" class="error-badge">
                                    (Real: {{ comparisonRight.true_label }})
                                </small>
                           </div>
                           
                           <table class="features-table">
                              <thead>
                                  <tr>
                                      <th>Sensor</th>
                                      <th>Media</th>
                                      <th>Std (Intensidad)</th>
                                      <th>Energía</th>
                                  </tr>
                              </thead>
                              <tbody>
                                  <tr v-for="channel in currentInspectionChannels" :key="channel.name"
                                      :class="getDiffClass(getFeatureValue(comparisonLeft.features, channel.idx, 0), getFeatureValue(comparisonRight.features, channel.idx, 0))">
                                      <td>{{ channel.name }}</td>
                                      <td>{{ getFeatureValue(comparisonRight.features, channel.idx, 0) }}</td>
                                      <td>{{ getFeatureValue(comparisonRight.features, channel.idx, 1) }}</td>
                                      <td>{{ getFeatureValue(comparisonRight.features, channel.idx, 6) }}</td>
                                  </tr>
                              </tbody>
                           </table>
                      </div>
                      
                      <div v-else class="empty-state">
                          <p>👈 Haz clic en otro segmento para comparar</p>
                      </div>
                  </div>
              </div>
           </div>
         </div>
      </div>

      <!-- Quick Guide Section for Non-Experts -->
      <div class="info-grid">
         <div class="card info-card">
           <h2>2. ¿Cómo funciona?</h2>
           <p>El modelo analiza <strong>2 segundos</strong> de movimiento a la vez.</p>
           <ul class="simple-list">
             <li>🏃‍♂️ <strong>Calcula Intensidad:</strong> Si los sensores agitan mucho -> Correr.</li>
             <li>🛌 <strong>Chequea Gravedad:</strong> Si el pecho apunta abajo -> Acostado.</li>
             <li>🔄 <strong>Busca Patrones:</strong> Balanceo rítmico de brazos -> Trotar.</li>
           </ul>
         </div>

         <!-- Confusion Matrix Section -->
         <div class="card matrix-card">
            <h2>3. Precisión del Modelo</h2>
            <p>Cuadros azules oscuros en la diagonal significan predicciones correctas.</p>
            <div class="image-container">
              <img src="/confusion_matrix.png" alt="Confusion Matrix" class="confusion-matrix-img" />
            </div>
         </div>
      </div>

    </main>
  </div>
</template>

<style scoped>
/* PREVIOS ESTILOS MANTENIDOS Y ACTUALIZADOS */
.inspection-panel {
    margin-top: 2rem;
    padding: 1.5rem;
    background: #fdfefe;
    border: 2px solid #bdc3c7;
    border-radius: 8px;
    animation: fadeIn 0.3s;
}

.panel-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1.5rem;
}

.header-controls {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}

.toggle-sensors {
    font-size: 0.95rem;
    color: #34495e;
    cursor: pointer;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.comparison-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2rem;
}

.data-column {
    background: white;
    padding: 1rem;
    border-radius: 8px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.05);
}

.column-header {
    display: flex;
    flex-direction: column;
    padding-bottom: 0.5rem;
    margin-bottom: 1rem;
}

.column-header strong { font-size: 1.3rem; }
.error-badge { color: #e74c3c; font-weight: bold; }

.features-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.9rem;
}

.features-table th, .features-table td {
    padding: 0.6rem;
    text-align: left;
    border-bottom: 1px solid #eee;
}

.features-table th { background-color: #f8f9fa; color: #7f8c8d; }

.empty-state {
    display: flex;
    align-items: center;
    justify-content: center;
    border: 2px dashed #bdc3c7;
    border-radius: 8px;
    color: #95a5a6;
    font-size: 1.2rem;
}

.close-btn {
    background: #e74c3c;
    padding: 0.5rem 1rem;
    font-size: 0.9rem;
}
.close-btn:hover { background: #c0392b; }

.timeline-segment { cursor: pointer; }
.timeline-segment.selected { 
    opacity: 1; 
    border: 2px solid #2c3e50; 
    z-index: 10; 
    transform: scaleY(1.3); 
    box-shadow: 0 0 5px rgba(0,0,0,0.5);
}

/* ERROR HIGHLIGHTING */
.timeline-segment.error-segment {
    filter: brightness(0.4) saturate(1.5); /* Mucho más oscuro */
    /* opcional: patrón de rayas si se pudiera, pero filter es suficiente */
}

.error-text { color: #e74c3c; }
.error-legend { font-size: 0.9rem; color: #666; display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1rem; }
.error-box-sample { width: 15px; height: 15px; background: #555; display: inline-block; }

@keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

/* RESTO DE ESTILOS (Actualizados) */
.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 2rem;
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  color: #333;
}
header { margin-bottom: 3rem; text-align: center; }
h1 { font-size: 2.5rem; color: #2c3e50; margin-bottom: 0.5rem; }
.subtitle { font-size: 1.2rem; color: #7f8c8d; }
.card { background: white; padding: 2rem; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); margin-bottom: 2rem; }
h2 { margin-bottom: 1rem; color: #2c3e50; font-size: 1.5rem; }
.instruction-text { color: #666; margin-bottom: 1.5rem; }
.upload-controls { display: flex; gap: 1rem; align-items: center; margin-bottom: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 8px; }
.summary-box { display: flex; justify-content: space-around; background: #ecf0f1; padding: 1.5rem; border-radius: 8px; margin-bottom: 2rem; }
.summary-item { display: flex; flex-direction: column; align-items: center; }
.summary-label { font-size: 0.9rem; color: #7f8c8d; text-transform: uppercase; letter-spacing: 1px; }
.summary-value { font-size: 1.8rem; font-weight: bold; color: #2c3e50; }
.timeline-vis { margin-top: 2rem; }
.chart-caption { text-align: center; font-style: italic; color: #95a5a6; margin-bottom: 1rem; }
.timeline-row { display: flex; align-items: center; margin-bottom: 1rem; }
.row-label { width: 120px; font-weight: bold; color: #34495e; }
.timeline-track { flex: 1; height: 30px; display: flex; background: #eee; border-radius: 4px; overflow: hidden; }
.timeline-segment:hover { opacity: 0.8; }
.legend { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 1rem; justify-content: center; }
.legend-item { display: flex; align-items: center; font-size: 0.8rem; }
.color-box { width: 15px; height: 15px; margin-right: 5px; border-radius: 3px; }
.info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
.simple-list { list-style: none; padding: 0; }
.simple-list li { margin-bottom: 1rem; font-size: 1.1rem; }
.image-container { display: flex; justify-content: center; padding: 1rem; }
.confusion-matrix-img { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid #eee; }
button { padding: 0.8rem 2rem; font-size: 1.1rem; background-color: #3498db; color: white; font-weight: bold; border: none; border-radius: 8px; cursor: pointer; transition: background 0.3s; }
button:hover { background-color: #2980b9; }
button:disabled { background-color: #95a5a6; cursor: not-allowed; }

.diff-high {
    background-color: #ffcccc; /* Light red for high difference */
    font-weight: bold;
}

.diff-med {
    background-color: #fff4cc; /* Light orange for medium difference */
}
</style>
```
