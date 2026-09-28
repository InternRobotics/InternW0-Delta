/* Editable metadata and media. Paper figures use the PDFs uploaded on 2026-09-24.
 * No author list, final release URL, citation, or experimental score is invented.
 */
window.WAM_CONTENT = {
  name: 'InternW0-Δ',
  title: 'An Embodied World Model Bridging Predictive Dynamics and Actions',
  subtitle: 'An Embodied World Model Bridging Predictive Dynamics and Actions',
  year: '2026',
  authors: [],
  affiliations: [],
  links: { paper: 'https://arxiv.org/abs/2609.31394', code: 'https://github.com/InternRobotics/InternW0-Delta', models: 'https://huggingface.co/collections/InternRobotics/internw0' },
  linkLabels: { paper: 'Read on arXiv ↗', code: 'View repository ↗', models: 'View collection ↗' },
  citation: '',
  hero: { title: 'Project video', subtitle: 'Real-robot demonstrations, model overview, and benchmark results.', src: 'assets/videos/web/main-20260927/main-demo.mp4', poster: 'assets/videos/posters/main-20260927/main-demo-cover.jpg' },
  figures: [
    {
      id: 'architecture', label: 'Architecture overview',
      image: 'assets/figures/architecture.png', pdf: 'assets/figures/architecture.pdf',
      source: 'assets/figures/architecture.pdf',
      alt: 'Original architecture figure: A/R/C observations pass through Wan VAE; T5 and proprioception form video context; the frozen VLM and proprioception form action context. Video and action experts interact through a 30-layer World Action MoT.'
    },
    {
      id: 'attention', label: 'World Action MoT and attention mask',
      image: 'assets/figures/attention.png', pdf: 'assets/figures/attention.pdf',
      source: 'assets/figures/attention.pdf',
      alt: 'Original MoT layer and mask: separate video and action projections, shared masked attention, expert-specific cross-attention and feed-forward layers. Delta attends to R, C, and Delta; Act attends to A, R, C, Delta, and Act, but not F.'
    },
    {
      id: 'causal-imprint', label: 'Causal Imprint supervision',
      image: 'assets/figures/causal-imprint.png', pdf: 'assets/figures/causal-imprint.pdf',
      source: 'assets/figures/causal-imprint.pdf',
      alt: 'Original Causal Imprint supervision diagram showing stop-gradient alignment from future-video features to delta features, and MSE supervision using ground-truth latent differences.'
    },
    {
      id: 'distillation', label: '4D-aware representation distillation',
      image: 'assets/figures/distillation.png', pdf: 'assets/figures/distillation.pdf',
      source: 'assets/figures/distillation.pdf',
      alt: 'Original 4D distillation diagram: offline Track4World geometry, 2D/3D motion, camera and visibility descriptors supervise a student branch reading clean A/R/C features from the 15th VideoDiT block through 16 learnable queries and a two-layer Transformer decoder.'
    }
  ],
  demos: [
    { id: 'grippers', label: 'Grippers', description: 'Everyday and laboratory tasks with gripper-equipped robots.', items: [
      { title: 'Filling a cup', subtitle: 'Filling a cup using a drink dispenser.', src: 'assets/videos/web/final-20260924/get-drink.mp4', poster: 'assets/videos/posters/final-20260924/get-drink.jpg' },
      { title: 'MOF experiment', subtitle: 'Transferring liquid in a MOF experiment.', src: 'assets/videos/web/final-20260924/mof-2000.mp4', poster: 'assets/videos/posters/final-20260924/mof-2000.jpg' },
      { title: 'Toasting bread', subtitle: 'Preparing toast with a bread toaster.', src: 'assets/videos/web/final-20260924/toast-bread-2000.mp4', poster: 'assets/videos/posters/final-20260924/toast-bread-2000.jpg' },
      { title: 'Luminol experiment', subtitle: 'Performing a luminol experiment.', src: 'assets/videos/web/final-20260924/luminol-2000.mp4', poster: 'assets/videos/posters/final-20260924/luminol-2000.jpg' },
      { title: 'Placing test tubes', subtitle: 'Placing test tubes into a rack.', src: 'assets/videos/web/final-20260924/insert-tubes-2000.mp4', poster: 'assets/videos/posters/final-20260924/insert-tubes-2000.jpg' }
    ]},
    { id: 'dexterous-hands', label: 'Dexterous hands', description: 'Manipulation with multi-fingered robotic hands.', items: [
      { title: 'Pouring water', subtitle: 'Pouring water with a dexterous hand.', src: 'assets/videos/web/final-20260924/pour-water-2000.mp4', poster: 'assets/videos/posters/final-20260924/pour-water-2000.jpg' },
      { title: 'Stacking paper cups', subtitle: 'Stacking paper cups with dexterous hands.', src: 'assets/videos/web/final-20260924/stack-cups-2000.mp4', poster: 'assets/videos/posters/final-20260924/stack-cups-2000.jpg' },
      { title: 'Dropper liquid transfer', subtitle: 'Transferring liquid with a dropper using a dexterous hand.', src: 'assets/videos/web/final-20260924/use-dropper-2000.mp4', poster: 'assets/videos/posters/final-20260924/use-dropper-2000.jpg' }
    ]}
  ],
  // Only user-approved examples and observed rejection reasons are published.
  // Four categories and eight newly supplied clips, September 22, 2026.
  // Preserve the user's definitions; do not infer the cause of an anomaly.
  dataFiltering: {
    categories: [
      { id: 'action_guard', treatment: 'Masked action targets', label: 'Implausible actions', description: 'Implausible actions within a trajectory segment. The affected action targets are masked while the video is retained.' },
      { id: 'motionless', label: 'Static segments', description: 'Segments containing static frames with no meaningful motion.' },
      { id: 'discontinuity', label: 'Recorded-signal discontinuities', description: 'Abrupt jumps in the recorded signals.' },
      { id: 'diverge', label: 'State–command mismatch', description: 'Inconsistency between the measured state and the commanded pose.' }
    ],
    examples: [
      { category: 'action_guard', title: 'Implausible actions · 1', label: 'Example 1', reason: 'Implausible actions within a trajectory segment. The affected action targets are masked while the video is retained.', src: 'assets/videos/filtering/web/action_guard_1.mp4', poster: 'assets/videos/filtering/posters/action_guard_1.jpg', width: 1920, height: 1080 },
      { category: 'action_guard', title: 'Implausible actions · 2', label: 'Example 2', reason: 'Implausible actions within a trajectory segment. The affected action targets are masked while the video is retained.', src: 'assets/videos/filtering/web/action_guard_2.mp4', poster: 'assets/videos/filtering/posters/action_guard_2.jpg', width: 1920, height: 1080 },
      { category: 'motionless', title: 'Static segments · 1', label: 'Example 1', reason: 'Segments containing static frames with no meaningful motion.', src: 'assets/videos/filtering/web/motionless_1.mp4', poster: 'assets/videos/filtering/posters/motionless_1.jpg', width: 1920, height: 1080 },
      { category: 'motionless', title: 'Static segments · 2', label: 'Example 2', reason: 'Segments containing static frames with no meaningful motion.', src: 'assets/videos/filtering/web/motionless_2.mp4', poster: 'assets/videos/filtering/posters/motionless_2.jpg', width: 1920, height: 1080 },
      { category: 'discontinuity', title: 'Recorded-signal discontinuities · 1', label: 'Example 1', reason: 'Abrupt jumps in the recorded signals.', src: 'assets/videos/filtering/web/discontinuity_1.mp4', poster: 'assets/videos/filtering/posters/discontinuity_1.jpg', width: 1920, height: 1080 },
      { category: 'discontinuity', title: 'Recorded-signal discontinuities · 2', label: 'Example 2', reason: 'Abrupt jumps in the recorded signals.', src: 'assets/videos/filtering/web/discontinuity_2.mp4', poster: 'assets/videos/filtering/posters/discontinuity_2.jpg', width: 1920, height: 1080 },
      { category: 'diverge', title: 'State–command mismatch · 1', label: 'Example 1', reason: 'Inconsistency between the measured state and the commanded pose.', src: 'assets/videos/filtering/web/diverge_1.mp4', poster: 'assets/videos/filtering/posters/diverge_1.jpg', width: 1920, height: 1080 },
      { category: 'diverge', title: 'State–command mismatch · 2', label: 'Example 2', reason: 'Inconsistency between the measured state and the commanded pose.', src: 'assets/videos/filtering/web/diverge_2.mp4', poster: 'assets/videos/filtering/posters/diverge_2.jpg', width: 1920, height: 1080 }
    ]
  },
  // Current results and paper comparison values supplied by the user,
  // Updated September 24, 2026; not independently verified. Only methods with supplied
  // results are plotted. RoboDojo score is omitted at the user's request.
  experiments: {
    // Official project/company marks are identifiers, not endorsements or score sources.
    methodMarks: {
      'Qwen-RobotManip': { src: 'assets/marks/qwen.png', source: 'https://github.com/QwenLM' },
      'Qwen-RobotManip-Context': { src: 'assets/marks/qwen.png', source: 'https://github.com/QwenLM' },
      'Π0.5': { src: 'assets/marks/physical-intelligence.png', source: 'https://www.pi.website/' },
      'OpenWAM-α': { src: 'assets/marks/openwam.png', source: 'https://openwam-official.github.io/' },
      'GPT-6-Astra': { src: 'assets/marks/openai.png', source: 'https://github.com/openai' },
      'InternW0-Δ': { src: 'assets/internw0-favicon.svg' }
    },
    note: 'All axes start at zero. RoboDojo uses a 0–30% range; the other panels use 0–100%.',
    benchmarks: [
      { id: 'libero-plus', title: 'LIBERO-Plus', setting: '', mark: { src: 'assets/marks/libero.png', wide: true, source: 'https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images/libero_logo.png' }, results: [
        { method: 'Π0.5', value: 84.4 },
        { method: 'OpenWAM-α', value: 69.2 },
        { method: 'Qwen-RobotManip-Context', value: 91.4 },
        { method: 'InternW0-Δ', value: 92.8 }
      ] },
      { id: 'robotwin-c2r', title: 'RoboTwin', setting: 'Clean2Random', mark: { src: 'assets/marks/robotwin.png', source: 'https://github.com/RoboTwin-Platform' }, results: [
        { method: 'Π0.5', value: 47.9 },
        { method: 'OpenWAM-α', value: 48.7 },
        { method: 'Qwen-RobotManip-Context', value: 69.4 },
        { method: 'InternW0-Δ', value: 71.9 }
      ] },
      { id: 'robodojo', title: 'RoboDojo', setting: '', axisMax: 30, mark: { src: 'assets/marks/robodojo.png', source: 'https://robodojo-benchmark.com/' }, results: [
        { method: 'Π0.5', value: 6.91 },
        { method: 'OpenWAM-α', value: 11.92 },
        { method: 'GPT-6-Astra', value: 22.48 },
        { method: 'InternW0-Δ', value: 23.9 }
      ] },
      { id: 'ebench', title: 'EBench', setting: '', results: [
        { method: 'Π0.5', value: 27.1 },
        { method: 'OpenWAM-α', value: 49.4 },
        { method: 'Qwen-RobotManip', value: 45.6 },
        { method: 'InternW0-Δ', value: 49.2 }
      ] },
      { id: 'robotwin-c2c', title: 'RoboTwin', setting: 'Clean2Clean', mark: { src: 'assets/marks/robotwin.png', source: 'https://github.com/RoboTwin-Platform' }, results: [
        { method: 'Π0.5', value: 73.1 },
        { method: 'OpenWAM-α', value: 89.4 },
        { method: 'Qwen-RobotManip-Context', value: 84.7 },
        { method: 'InternW0-Δ', value: 90.0 }
      ] }
    ]
  }
};
