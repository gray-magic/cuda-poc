//using OpenGL;
using OpenCL.Net;
using System;
using System.Collections.Generic;
using System.Data;
using System.Linq;
using System.Windows.Forms;

namespace Test_GPU
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            try
            {


                //https://stackoverflow.com/questions/5654048/complete-net-opencl-implementations
                //https://www.codeproject.com/Articles/1116907/How-to-Use-Your-GPU-in-NET
                //https://stackoverflow.com/questions/46392132/c-sharp-opencl-gpu-implementation-for-double-array-math

                Find_Devices();
            }
            catch (Exception)
            {

                throw;
            }
        }

        private ErrorCode errorCode;
        private List<Device> Lst_Devices = new List<Device>();

        //Rechercher les GPU compatibles CUDA 
        private void Find_Devices()
        {
            try
            {
                //Collecter les equipements 
                Platform[] platforms = Cl.GetPlatformIDs(out errorCode);

                foreach (Platform platform in platforms)
                {

                    string platformName = Cl.GetPlatformInfo(platform, PlatformInfo.Name, out errorCode).ToString();
                    Lsb_Devices.Items.Add(platformName);

                    //GPU compatible uniquement 
                    foreach (Device device in Cl.GetDeviceIDs(platform, DeviceType.Gpu, out errorCode))
                    {
                        Lst_Devices.Add(device);
                    }
                }

                if (Lst_Devices.Count <= 0)
                {
                    throw new Exception("Aucun GPU CUDA");
                }
            }
            catch (Exception ex)
            {

                MessageBox.Show(ex.Message);
            }
        }


        //Choix du GPU 
        Device Gpu_1;
        private void Lsb_Devices_SelectedIndexChanged(object sender, EventArgs e)
        {
            try
            {
                Gpu_1 = Lst_Devices[Lsb_Devices.SelectedIndex];
            }
            catch (Exception ex)
            {

                MessageBox.Show(ex.Message);
            }
        }



        //Exécution mono GPU 
        private void Execute_GPU()
        {
            try
            {
                //Fonctionnement général 
                //Détécter les cartes graphiques compatibles CUDA
                //Définir la ou les cartes à utiliser 
                //Créer un contexte
                //Créer une file de commandes
                //Allouer et initialiser la mémoire du device
                //Charger et compiler le kernel
                //Insérer le kernel dans la file de commandes
                //Récupérer les données dans la mémoire du device
                //Libérer les ressources

                //Création du contexte (suivant liste des GPU)
                Context Gpu_context = Cl.CreateContext(null, 1, new Device[] { Gpu_1 }, null, IntPtr.Zero, out errorCode);
                if (errorCode != ErrorCode.Success)
                {
                    throw new Exception("Impossible de créer le contexte");
                }

                CommandQueue commandQueue = Cl.CreateCommandQueue(Gpu_context, Gpu_1, CommandQueueProperties.OutOfOrderExecModeEnable, out errorCode);
                if (errorCode != ErrorCode.Success)
                {
                    throw new Exception("Impossible de créer la liste de traitement");
                }


                Event event0;
                ErrorCode err;
                String Function_Name = "My_Function";

                //Programme à éxécuter en OpenCL-C 
                List<String> Lst_Row_Code_Build = new List<string>();
                Lst_Row_Code_Build.Add(" __kernel void " + Function_Name + "(__global float* input, __global float* output) ");
                Lst_Row_Code_Build.Add("{");
                Lst_Row_Code_Build.Add("size_t i = get_global_id(0);");
                Lst_Row_Code_Build.Add(" output[i] = input[i] + input[i];");
                Lst_Row_Code_Build.Add("};");
                String Program_Source_Code = String.Join(System.Environment.NewLine, Lst_Row_Code_Build);

                //Création du programme 
                OpenCL.Net.Program program = Cl.CreateProgramWithSource(Gpu_context, 1, new[] { Program_Source_Code }, null, out err);
                Cl.BuildProgram(program, 0, null, string.Empty, null, IntPtr.Zero);

                //Récupérations des informations du build
                if (Cl.GetProgramBuildInfo(program, Gpu_1, ProgramBuildInfo.Status, out err).CastTo<BuildStatus>() != BuildStatus.Success)
                {
                    //Affichage si erreurs durant la compilation 
                    if (err != ErrorCode.Success)
                    {
                        Txb_Output.Text += String.Format("ERROR: " + "Cl.GetProgramBuildInfo" + " (" + err.ToString() + ")");
                        Txb_Output.Text += String.Format("Cl.GetProgramBuildInfo != Success");
                        Txb_Output.Text += Cl.GetProgramBuildInfo(program, Gpu_1, ProgramBuildInfo.Log, out err);
                    }
                }

                // Création d'un noyaux pour le programme 
                OpenCL.Net.Kernel kernel = Cl.CreateKernel(program, Function_Name, out err);

                // Allouer des tampons d'entrée et de sortie et remplir l'entrée avec des données
                const int count = 2048;
                Mem memInput = (Mem)Cl.CreateBuffer(Gpu_context, MemFlags.ReadOnly, sizeof(float) * count, out err);

                // Créer une mémoire tampon de sortie pour les résultats
                Mem memoutput = (Mem)Cl.CreateBuffer(Gpu_context, MemFlags.WriteOnly, sizeof(float) * count, out err);

                // Génération des données de tests aléatoires
                var random = new Random();
                float[] data = (from i in Enumerable.Range(0, count) select (float)random.NextDouble()).ToArray();

                //Copier le tampon hôte de valeurs de tests aléatoires dans le tampon du périphérique d'entrée
                Cl.EnqueueWriteBuffer(commandQueue, (IMem)memInput, Bool.True, IntPtr.Zero, new IntPtr(sizeof(float) * count), data, 0, null, out event0);

                //Utiliser le nombre maximal d'éléments de travail pris en charge pour ce noyau sur ce périphérique
                IntPtr notused;
                InfoBuffer local = new InfoBuffer(new IntPtr(4));
                Cl.GetKernelWorkGroupInfo(kernel, Gpu_1, KernelWorkGroupInfo.WorkGroupSize, new IntPtr(sizeof(int)), local, out notused);

                //Définission des arguments du noyau et mise en file d'attente pour éxécution
                Cl.SetKernelArg(kernel, 0, new IntPtr(4), memInput);
                Cl.SetKernelArg(kernel, 1, new IntPtr(4), memoutput);
                Cl.SetKernelArg(kernel, 2, new IntPtr(4), count);
                IntPtr[] workGroupSizePtr = new IntPtr[] { new IntPtr(count) };
                Cl.EnqueueNDRangeKernel(commandQueue, kernel, 1, null, workGroupSizePtr, null, 0, null, out event0);

                //Forcer le traitement de la file d'attente de commandes, attendre que toutes les commandes soient terminées
                //clFinish (le host attend la fin de la file)
                //clWaitForEvent (le host attend la fin d'une commande)
                //clEnqueueBarrier (le device attend la fin des commandes antérieures)
                //clEnqueueWaitForEvents(le device attend la fin d'une commande)
                Cl.Finish(commandQueue);

                //Lecture/récupération des résultats 
                float[] results = new float[count];
                Cl.EnqueueReadBuffer(commandQueue, (IMem)memoutput, Bool.True, IntPtr.Zero, new IntPtr(count * sizeof(float)), results, 0, null, out event0);

                //Validation des résultats 
                int correct = 0;
                for (int i = 0; i < count; i++)
                    correct += (results[i] == data[i] + data[i]) ? 1 : 0;

                //Retour utilisateur 
                Txb_Output.Text = String.Format("Computed {0} of {1} correct values!", correct.ToString(), count.ToString());

            }
            catch (Exception)
            {

                throw;
            }
        }


        private void Btn_Execute_Click(object sender, EventArgs e)
        {
            try
            {
                Execute_GPU();
            }
            catch (Exception ex)
            {

                MessageBox.Show(ex.Message);
            }
        }
    }


}
