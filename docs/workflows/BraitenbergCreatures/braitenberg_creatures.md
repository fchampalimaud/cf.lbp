# Braitenberg vehicles


## Details
1. Creates a connection with the Behavior board. The PortName property in the Behavior node needs to be set to the COM device on the computer. 
2. Filters the messages from the Behavior board that pertain analog inputs.
3. Selects the AnalogInput0 (AD0) from the list of possible analog inputs (see the output of the Parse node in 2).


<div style="display: flex; justify-content: center; gap: 40px;">
  <figure style="text-align: center; transform: rotate(20deg);">
    <img src="Braitenbergs1.svg" alt="Braitenberg1" width="45%">
    <figcaption>Caption for Image 1</figcaption>
  </figure>
</div>


<div style="display: flex; justify-content: center; gap: 40px;">
  <figure style="text-align: center; transform: rotate(-20deg);">
    <img src="Braitenbergs2a.svg" alt="Braitenberg2a" width="45%">
    <figcaption>Caption for Image 1</figcaption>
  </figure>
  <figure style="text-align: center; transform: rotate(20deg);">
    <img src="Braitenbergs2b.svg" alt="Braitenberg2b" width="45%">
    <figcaption>Caption for Image 2</figcaption>
  </figure>
</div>

<div style="display: flex; justify-content: center; gap: 40px;">
  <figure style="text-align: center; transform: rotate(-20deg);">
    <img src="Braitenbergs3a.svg" alt="Braitenberg3a" width="45%">
    <figcaption>Caption for Image 1</figcaption>
  </figure>
  <figure style="text-align: center; transform: rotate(20deg);">
    <img src="Braitenbergs3b.svg" alt="Braitenberg3b" width="45%">
    <figcaption>Caption for Image 2</figcaption>
  </figure>
</div>



<!--
![Schematics](./Braitenbergs2a.svg){ height=2% }

![Schematics](./Braitenbergs1.svg){ height=2% }
-->



## Workflow
:::workflow
![Example](~/workflows/HarpExamples/BehaviorBoard/AnalogInput/AnalogInput.bonsai)
:::
