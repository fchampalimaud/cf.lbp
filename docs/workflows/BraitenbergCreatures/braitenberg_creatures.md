# Braitenberg vehicles


## Details
1. Creates a connection with the Behavior board. The PortName property in the Behavior node needs to be set to the COM device on the computer. 
2. Filters the messages from the Behavior board that pertain analog inputs.
3. Selects the AnalogInput0 (AD0) from the list of possible analog inputs (see the output of the Parse node in 2).


<div style="display: flex; justify-content: center; gap: 10px;">
    <img src="Braitenbergs1.svg" alt="Braitenberg1" style="transform: rotate(10deg); height: 200px;">
</div>
<p style="text-align: center; font-size: 18px; font-weight: bold; margin-top: 20px;">Figure1: caption for Braitenberg1</p>

<div style="display: flex; justify-content: center; gap: 50px;">
    <img src="Braitenbergs2a.svg" alt="Braitenberg2a" style="transform: rotate(-10deg); height: 200px;margin-right: 100px;">
    <img src="Braitenbergs2b.svg" alt="Braitenberg2b" style="transform: rotate(10deg); height: 200px;">
</div>
<p style="text-align: center; font-size: 18px; font-weight: bold; margin-top: 20px;">Figure2: caption for Braitenbergs2a and Braitenbergs2b</p>

<div style="display: flex; justify-content: center; gap: 30px;">
    <img src="Braitenbergs3a.svg" alt="Braitenberg3a" style="transform: rotate(-10deg); height: 200px;margin-right: 100px;">
    <img src="Braitenbergs3b.svg" alt="Braitenberg3b" style="transform: rotate(10deg); height: 200px;">
</div>
<p style="text-align: center; font-size: 18px; font-weight: bold; margin-top: 20px;">Figure3: caption for Braitenbergs3a and Braitenbergs3b</p>

<!--
![Schematics](./Braitenbergs2a.svg){ height=2% }

![Schematics](./Braitenbergs1.svg){ height=2% }
-->



## Workflow
:::workflow
![Example](~/workflows/HarpExamples/BehaviorBoard/AnalogInput/AnalogInput.bonsai)
:::
